# Архитектура дискретно-событийной симуляции мультиагентной системы

**Автор:** Principal Systems Architect & Simulation Engineer  
**Дата:** 14 октября 2025  
**Статус:** Design & Planning Phase

---

## КЛЮЧЕВОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ

**EventQueue хранит только события START (REQUEST_ARRIVAL, TOOL_START)**

Вместо традиционного подхода с тремя типами событий (ARRIVAL, START, FINISH), используется упрощённая модель:

✅ **Что в очереди:** Только запланированные старты задач  
✅ **Где finish times:** Вычисляются динамически на каждом шаге  
✅ **Как работает:**
1. На каждом шаге находим ближайшее завершение среди активных задач
2. Сравниваем с ближайшим стартом из EventQueue
3. Продвигаем время в `min(next_start, next_finish)`
4. Обновляем `remaining_work` для всех активных задач пропорционально прошедшему времени

**Преимущества:**
- Не нужно удалять/перепланировать FINISH события при изменении resource sharing
- 100% точность (всегда используем актуальное состояние системы)
- Меньше событий в очереди
- Явный контроль над временем симуляции

**Сложность:** O(N_active × N_resources) на каждый шаг (неизбежно при dynamic fair-share)

---

## 1. EXECUTIVE SUMMARY

### Цели симуляции
1. **Прямая задача**: Оценить среднюю latency запросов при заданном throughput (запросов/мин)
2. **Обратная задача**: Оценить максимальный throughput при ограничении по latency SLA

### Ключевые метрики
- **Latency**: время от поступления запроса до завершения всех инструментов в его DAG
- **Throughput**: количество успешно обработанных запросов за единицу времени
- **Resource Utilization**: процент использования каждого типа ресурса (CPU/NPU/Memory/Network/Disk)
- **Queue Depth**: количество ожидающих/активных инструментов в каждый момент времени

---

## 2. КОНЦЕПТУАЛЬНАЯ МОДЕЛЬ

### 2.1 Основные сущности

```
Request (Запрос)
├── request_id: UUID
├── request_type: str (web-search, product-matching, deep-research, etc.)
├── arrival_time: float (секунды от начала симуляции)
├── dag: AgenticToolGraph
├── start_time: Optional[float]
├── finish_time: Optional[float]
└── tool_instances: Dict[str, ToolInstance]

ToolInstance (Экземпляр инструмента для конкретного запроса)
├── tool_id: str (уникальный: f"{request_id}_{tool_name}")
├── tool_name: str
├── request_id: UUID
├── tool_template: AgenticTool (ссылка на шаблон из конфига)
├── status: ToolStatus (PENDING, RUNNING, COMPLETED)
├── dependencies: List[str] (имена инструментов в DAG)
├── dependents: List[str] (инструменты, ждущие завершения этого)
├── start_time: Optional[float]
├── finish_time: Optional[float]
├── allocated_share: Dict[str, float] (доля каждого ресурса)
└── remaining_work: Dict[str, float] (оставшаяся работа по каждому ресурсу)

Resource (Ресурс)
├── resource_type: ResourceType (CPU, NPU, MEMORY, NETWORK, DISK)
├── total_capacity: float
├── available_capacity: float
└── active_consumers: Set[str] (множество tool_id)

Event (Событие)
├── event_id: int
├── event_type: EventType (REQUEST_ARRIVAL, TOOL_START, TOOL_FINISH)
├── timestamp: float
├── payload: Dict[str, Any]
└── priority: int (для событий с одинаковым timestamp)
```

### 2.2 Типы событий

**В EventQueue хранятся только события старта задач:**

| Тип события | Описание | Payload | Триггеры |
|-------------|----------|---------|----------|
| `REQUEST_ARRIVAL` | Поступление нового запроса в систему | `{request_id, request_type}` | Генерируется заранее по Poisson процессу |
| `TOOL_START` | Начало выполнения инструмента | `{tool_id, request_id, tool_name}` | Завершение всех dependencies OR REQUEST_ARRIVAL для root tools |

**Важно:** События завершения (`TOOL_FINISH`) НЕ хранятся в очереди. Вместо этого на каждом шаге симуляции мы:
1. Вычисляем ближайший момент завершения работы по любому ресурсу для любой активной задачи
2. Сравниваем с ближайшим `TOOL_START` из EventQueue
3. Продвигаем время в минимум из двух значений

---

## 3. АЛГОРИТМЫ И ФОРМУЛЫ

### 3.1 Генерация запросов (Arrival Process)

**Вариант 1: Детерминированный поток**
```
Inter-arrival time = 60 / λ  (секунды)
где λ = заданный throughput (запросов/мин)

Преимущества:
+ Простая реализация
+ Предсказуемое поведение
+ Детерминированный worst-case анализ

Недостатки:
- Не отражает реальную вариативность
- Может недооценивать queueing effects
```

**Вариант 2: Poisson процесс** ✓ (рекомендуется)
```
Inter-arrival time ~ Exponential(λ)
T_arrival[i+1] = T_arrival[i] + Exp(1/λ)

где λ = средний throughput (запросов/мин) / 60

Преимущества:
+ Отражает реальную стохастическую природу
+ Моделирует burst behavior
+ Классическая модель для queueing theory

Недостатки:
- Требует генератор случайных чисел
- Результаты требуют усреднения по multiple runs
```

**Формула генерации:**
```python
import numpy as np

def generate_arrival_times(lambda_per_min: float, simulation_duration: float) -> List[float]:
    """
    Генерирует времена поступления запросов по Poisson процессу.
    
    Args:
        lambda_per_min: средний throughput (запросов/мин)
        simulation_duration: длительность симуляции (секунды)
    
    Returns:
        Список временных меток arrival в секундах
    """
    lambda_per_sec = lambda_per_min / 60.0
    arrival_times = []
    current_time = 0.0
    
    while current_time < simulation_duration:
        inter_arrival = np.random.exponential(1.0 / lambda_per_sec)
        current_time += inter_arrival
        if current_time < simulation_duration:
            arrival_times.append(current_time)
    
    return arrival_times
```

### 3.2 Fair Share распределение ресурсов

**Базовый принцип:** В каждый момент времени `t`, если `N` активных инструментов используют ресурс `R` с total capacity `C_R`, каждый получает долю `C_R / N`.

**Формула для вычисления доли:**

```
Share[i, r, t] = C_r / N_r(t)

где:
- C_r = total capacity ресурса r
- N_r(t) = количество активных инструментов, использующих ресурс r в момент t
```

**Ключевая идея:** Мы НЕ храним запланированные времена завершения. Вместо этого на каждом шаге вычисляем:
1. Для каждой активной задачи и каждого используемого ресурса: когда завершится работа по этому ресурсу
2. Находим минимум = момент первого завершения

### 3.3 Алгоритм определения следующего момента симуляции

**Основной цикл симуляции:**

```python
def find_next_completion(current_time: float, active_tools: Set[ToolInstance]) -> Tuple[float, str, ResourceType]:
    """
    Находит ближайший момент завершения работы по любому ресурсу для любой активной задачи.
    
    Returns:
        (completion_time, tool_id, resource_type) - момент завершения, задача, ресурс
    """
    min_completion_time = float('inf')
    completing_tool = None
    completing_resource = None
    
    # Подсчитать количество потребителей для каждого ресурса
    resource_consumers = defaultdict(set)
    for tool in active_tools:
        for resource_type in ResourceType:
            if tool.remaining_work[resource_type] > 0:
                resource_consumers[resource_type].add(tool.tool_id)
    
    # Для каждой задачи и каждого ресурса вычислить время завершения
    for tool in active_tools:
        for resource_type in ResourceType:
            remaining = tool.remaining_work[resource_type]
            
            if remaining <= 0:
                continue  # Работа по этому ресурсу уже завершена
            
            # Fair share для этого ресурса
            num_consumers = len(resource_consumers[resource_type])
            share = resources[resource_type].total_capacity / num_consumers
            
            # Время завершения работы по этому ресурсу
            completion_time = current_time + (remaining / share)
            
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                completing_tool = tool
                completing_resource = resource_type
    
    return min_completion_time, completing_tool, completing_resource


def simulation_step():
    """
    Один шаг симуляции: определяет следующий момент времени и обрабатывает событие.
    """
    # 1. Найти ближайший старт из EventQueue
    next_start_time = event_queue.peek().timestamp if event_queue else float('inf')
    
    # 2. Найти ближайшее завершение среди активных задач
    next_completion_time, completing_tool, completing_resource = find_next_completion(
        current_time, active_tools
    )
    
    # 3. Выбрать минимум
    next_time = min(next_start_time, next_completion_time)
    
    if next_time == float('inf'):
        # Нет больше событий - симуляция завершена
        return False
    
    # 4. Продвинуть время
    time_delta = next_time - current_time
    current_time = next_time
    
    # 5. Обработать событие
    if next_time == next_start_time:
        # Событие старта задачи
        event = event_queue.pop()
        handle_tool_start(event)
    else:
        # Завершение работы по ресурсу
        handle_resource_completion(completing_tool, completing_resource, time_delta)
    
    return True
```

**Сложность:** O(|ActiveTools| × |ResourceTypes|) на каждый шаг  
**Плюсы:** 
- Не нужно хранить и перепланировать TOOL_FINISH события
- Всегда актуальные вычисления с учётом текущего состояния
- Явный контроль над временем симуляции

**Минусы:** 
- Вычисления на каждом шаге (но это неизбежно при dynamic resource sharing)

### 3.3.1 Пример работы алгоритма

**Сценарий:** Два инструмента (Tool A и Tool B) стартуют одновременно.

```
Начальное состояние (t=0):
- Tool A: needs 100 units of CPU, 50 units of Network
- Tool B: needs 80 units of CPU
- Resources: CPU capacity = 100, Network capacity = 100

EventQueue: [START(A, t=0), START(B, t=0)]
Active: []
```

**Шаг 1: t=0, обработка START(A)**
```
Active: [A]
Remaining work: A = {CPU: 100, Network: 50}

Find next completion:
- A on CPU: 0 + 100/100 = 1.0 sec
- A on Network: 0 + 50/100 = 0.5 sec
=> Next completion: t=0.5 (A finishes Network)

Next event: min(START(B, t=0), completion=0.5) = 0
=> Process START(B)
```

**Шаг 2: t=0, обработка START(B)**
```
Active: [A, B]
Remaining work: A = {CPU: 100, Network: 50}, B = {CPU: 80, Network: 0}

Find next completion:
- A on CPU: 0 + 100/50 = 2.0 sec  (fair-share: 100/2 consumers)
- A on Network: 0 + 50/100 = 0.5 sec
- B on CPU: 0 + 80/50 = 1.6 sec
=> Next completion: t=0.5 (A finishes Network)

Next event: min(EventQueue=empty, completion=0.5) = 0.5
=> Process completion at t=0.5
```

**Шаг 3: t=0.5, A завершает Network**
```
Update remaining work (time_delta = 0.5):
- A on CPU: work_done = 50 * 0.5 = 25 => remaining = 100 - 25 = 75
- A on Network: work_done = 100 * 0.5 = 50 => remaining = 0 ✓ (завершено)
- B on CPU: work_done = 50 * 0.5 = 25 => remaining = 80 - 25 = 55

Active: [A, B]  (A всё ещё активна, работает над CPU)
Remaining work: A = {CPU: 75, Network: 0}, B = {CPU: 55, Network: 0}

Find next completion:
- A on CPU: 0.5 + 75/50 = 2.0 sec
- B on CPU: 0.5 + 55/50 = 1.6 sec
=> Next completion: t=1.6 (B finishes CPU)
```

**Шаг 4: t=1.6, B завершает CPU**
```
Update remaining work (time_delta = 1.1):
- A on CPU: work_done = 50 * 1.1 = 55 => remaining = 75 - 55 = 20
- B on CPU: work_done = 50 * 1.1 = 55 => remaining = 0 ✓

B полностью завершена (все ресурсы = 0)
Active: [A]
Remaining work: A = {CPU: 20, Network: 0}

Find next completion:
- A on CPU: 1.6 + 20/100 = 1.8 sec  (теперь A одна, получает 100%)
=> Next completion: t=1.8
```

**Шаг 5: t=1.8, A завершает CPU**
```
Update remaining work (time_delta = 0.2):
- A on CPU: work_done = 100 * 0.2 = 20 => remaining = 0 ✓

A полностью завершена
Active: []

Результат:
- Tool A latency: 1.8 sec
- Tool B latency: 1.6 sec
```

**Проверка корректности:**
- Tool A: CPU работа = 25 (fair-share @ 0.5s) + 55 (fair-share @ 1.1s) + 20 (100% @ 0.2s) = 100 ✓
- Tool B: CPU работа = 25 (fair-share @ 0.5s) + 55 (fair-share @ 1.1s) = 80 ✓
- Network: Tool A получила 50 за 0.5s @ 100% capacity ✓

### 3.4 Обработка завершения работы по ресурсу

Когда задача завершает работу по ресурсу, нужно:
1. Обновить `remaining_work` для ВСЕХ активных задач (так как прошло время `time_delta`)
2. Проверить, не завершилась ли задача полностью (все ресурсы)
3. Если завершилась — проверить зависимости и запланировать старты

```python
def handle_resource_completion(tool: ToolInstance, resource: ResourceType, time_delta: float) -> None:
    """
    Обрабатывает завершение работы по одному ресурсу для одной задачи.
    
    Args:
        tool: задача, которая завершила работу по ресурсу
        resource: ресурс, по которому завершена работа
        time_delta: прошедшее время с предыдущего шага
    """
    # 1. Обновить remaining_work для ВСЕХ активных задач
    #    (так как за time_delta каждая задача выполнила некоторую работу)
    resource_consumers = defaultdict(set)
    for active_tool in active_tools:
        for r in ResourceType:
            if active_tool.remaining_work[r] > 0:
                resource_consumers[r].add(active_tool.tool_id)
    
    for active_tool in active_tools:
        for r in ResourceType:
            if active_tool.remaining_work[r] > 0:
                # Вычислить fair share для этого ресурса
                num_consumers = len(resource_consumers[r])
                share = resources[r].total_capacity / num_consumers
                
                # Выполненная работа за time_delta
                work_done = share * time_delta
                
                # Обновить remaining_work (не может быть < 0)
                active_tool.remaining_work[r] = max(0, active_tool.remaining_work[r] - work_done)
    
    # 2. Проверить, завершилась ли задача tool полностью
    if all(tool.remaining_work[r] <= 1e-9 for r in ResourceType):  # epsilon для float
        tool.status = ToolStatus.COMPLETED
        tool.finish_time = current_time
        active_tools.remove(tool)
        
        # 3. Проверить зависимости и запустить dependent tasks
        check_and_start_dependents(tool)


def check_and_start_dependents(finished_tool: ToolInstance) -> None:
    """
    Проверяет, можно ли запустить задачи, зависящие от finished_tool,
    и добавляет события TOOL_START в EventQueue.
    """
    request = requests[finished_tool.request_id]
    dag = request.dag
    
    # Найти всех dependents в DAG
    for dependent_name in dag.graph.successors(finished_tool.tool_name):
        dependent_tool = request.tool_instances[dependent_name]
        
        # Проверить, все ли dependencies завершены
        all_deps_done = all(
            request.tool_instances[dep_name].status == ToolStatus.COMPLETED
            for dep_name in dag.graph.predecessors(dependent_name)
        )
        
        if all_deps_done and dependent_tool.status == ToolStatus.PENDING:
            # Добавить событие TOOL_START в очередь
            event = Event(
                event_type=EventType.TOOL_START,
                timestamp=current_time,  # старт немедленно (или можно добавить delay)
                payload={'tool_id': dependent_tool.tool_id}
            )
            event_queue.push(event)
    
    # Проверить, завершён ли весь запрос
    if all(t.status == ToolStatus.COMPLETED for t in request.tool_instances.values()):
        request.finish_time = current_time
        # Записать метрику latency
        latency = request.finish_time - request.arrival_time
        metrics.record_request_latency(request.request_type, latency)
```

---

## 4. АРХИТЕКТУРА КОМПОНЕНТОВ

### 4.1 Структура модулей

```
mksim/
├── agentic/
│   ├── tool/
│   │   ├── tools.py               # (существующие классы AgenticTool)
│   │   ├── tool_factory.py        # (существующая фабрика)
│   │   └── tool_instance.py       # [NEW] Экземпляр инструмента для запроса
│   │
│   └── request/
│       ├── request.py             # [NEW] Класс Request
│       └── request_generator.py   # [NEW] Генератор запросов по Poisson
│
├── simulator/
│   ├── event.py                   # [NEW] Event, EventType, EventQueue
│   ├── resource.py                # [NEW] Resource, ResourceManager
│   └── simulation_engine.py       # [NEW] Основной цикл симуляции (включает fair-share logic)
│
├── metrics/
│   ├── collector.py               # [NEW] Сбор метрик (latency, throughput, utilization)
│   └── analyzer.py                # [NEW] Анализ результатов, percentiles, SLA compliance
│
└── experiments/
    ├── scenario.py                # [NEW] Определение сценария эксперимента
    └── runner.py                  # [NEW] Запуск multiple runs, агрегация результатов

utils/
├── agentic_tool_graph.py          # (существующий)
└── config_iterator.py             # (существующий)

configs/
├── config.yaml                    # (существующий)
├── scenarios/                     # [NEW] Сценарии экспериментов
│   ├── web_search_load_test.yaml
│   ├── deep_research_sla.yaml
│   └── mixed_workload.yaml
└── resources/                     # [NEW] Конфигурация системных ресурсов
    └── cluster_spec.yaml

main.py                            # (существующий, адаптировать)
simulation_runner.py               # [NEW] Entry point для запуска симуляции
```

### 4.2 Класс SimulationEngine (ядро)

```python
from heapq import heappush, heappop
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class EventType(Enum):
    REQUEST_ARRIVAL = 1
    TOOL_START = 2

@dataclass
class Event:
    timestamp: float
    event_type: EventType
    payload: Dict
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

class SimulationEngine:
    """
    Дискретно-событийный симулятор с fair-share resource allocation.
    
    Основная идея: в EventQueue хранятся только события старта (REQUEST_ARRIVAL, TOOL_START).
    Время завершения задач вычисляется динамически на каждом шаге.
    """
    
    def __init__(self, 
                 resource_manager: ResourceManager,
                 metrics_collector: MetricsCollector):
        self.event_queue: List[Event] = []  # heapq для стартов
        self.current_time: float = 0.0
        self.resource_manager = resource_manager
        self.metrics = metrics_collector
        
        self.requests: Dict[UUID, Request] = {}
        self.active_tools: Set[ToolInstance] = set()
        
    def schedule_event(self, event: Event) -> None:
        """Добавляет событие в очередь."""
        heappush(self.event_queue, event)
    
    def run(self, until: float) -> None:
        """
        Основной цикл симуляции.
        
        Args:
            until: время окончания симуляции (секунды)
        """
        while True:
            # 1. Найти ближайший старт из EventQueue
            next_start_time = self.event_queue[0].timestamp if self.event_queue else float('inf')
            
            # Проверка на окончание симуляции
            if next_start_time > until and not self.active_tools:
                break
            
            # 2. Найти ближайшее завершение среди активных задач
            next_completion_time, completing_tool, completing_resource = self._find_next_completion()
            
            # 3. Определить следующий момент времени
            next_time = min(next_start_time, next_completion_time)
            
            if next_time == float('inf'):
                # Нет больше событий
                break
            
            if next_time > until:
                # Симуляция завершена
                break
            
            # 4. Обработать событие
            if next_time == next_start_time:
                # Событие старта
                event = heappop(self.event_queue)
                self.current_time = event.timestamp
                
                if event.event_type == EventType.REQUEST_ARRIVAL:
                    self._handle_request_arrival(event)
                elif event.event_type == EventType.TOOL_START:
                    self._handle_tool_start(event)
            else:
                # Завершение работы по ресурсу
                time_delta = next_completion_time - self.current_time
                self.current_time = next_completion_time
                self._handle_resource_completion(completing_tool, completing_resource, time_delta)
            
            # После каждого шага обновляем метрики
            self.metrics.snapshot(self.current_time, 
                                  self.active_tools, 
                                  self.resource_manager)
    
    def _find_next_completion(self) -> Tuple[float, Optional[ToolInstance], Optional[ResourceType]]:
        """
        Находит ближайший момент завершения работы по любому ресурсу для любой активной задачи.
        
        Returns:
            (completion_time, tool, resource_type) или (inf, None, None) если нет активных задач
        """
        if not self.active_tools:
            return float('inf'), None, None
        
        min_completion_time = float('inf')
        completing_tool = None
        completing_resource = None
        
        # Подсчитать количество потребителей для каждого ресурса (для fair-share)
        resource_consumers = defaultdict(set)
        for tool in self.active_tools:
            for resource_type in ResourceType:
                if tool.remaining_work[resource_type] > 1e-9:  # epsilon
                    resource_consumers[resource_type].add(tool.tool_id)
        
        # Для каждой задачи и каждого ресурса вычислить время завершения
        for tool in self.active_tools:
            for resource_type in ResourceType:
                remaining = tool.remaining_work[resource_type]
                
                if remaining <= 1e-9:  # epsilon для float сравнения
                    continue
                
                # Fair share для этого ресурса
                num_consumers = len(resource_consumers[resource_type])
                if num_consumers == 0:
                    continue
                    
                capacity = self.resource_manager.get_capacity(resource_type)
                share = capacity / num_consumers
                
                # Время завершения работы по этому ресурсу
                completion_time = self.current_time + (remaining / share)
                
                if completion_time < min_completion_time:
                    min_completion_time = completion_time
                    completing_tool = tool
                    completing_resource = resource_type
        
        return min_completion_time, completing_tool, completing_resource
    
    def _handle_request_arrival(self, event: Event) -> None:
        """Обработка поступления нового запроса."""
        request = self._create_request(event.payload)
        self.requests[request.request_id] = request
        
        # Найти root tools (без dependencies)
        dag = request.dag
        root_tools = [node for node in dag.graph.nodes 
                      if dag.graph.in_degree(node) == 0]
        
        # Создать события TOOL_START для root tools
        for tool_name in root_tools:
            tool_instance = request.tool_instances[tool_name]
            start_event = Event(
                timestamp=self.current_time,
                event_type=EventType.TOOL_START,
                payload={'tool_id': tool_instance.tool_id}
            )
            self.schedule_event(start_event)
    
    def _handle_tool_start(self, event: Event) -> None:
        """Обработка начала выполнения инструмента."""
        tool_id = event.payload['tool_id']
        # Найти tool instance по ID (из requests)
        tool = self._find_tool_by_id(tool_id)
        
        tool.status = ToolStatus.RUNNING
        tool.start_time = self.current_time
        
        # Добавить в активные задачи
        self.active_tools.add(tool)
        
        # Инициализировать remaining_work
        for resource_type in ResourceType:
            tool.remaining_work[resource_type] = tool.get_load(resource_type)
    
    def _handle_resource_completion(self, tool: ToolInstance, resource: ResourceType, time_delta: float) -> None:
        """
        Обрабатывает завершение работы по одному ресурсу для одной задачи.
        
        Args:
            tool: задача, которая завершила работу по ресурсу
            resource: ресурс, по которому завершена работа
            time_delta: прошедшее время с предыдущего шага
        """
        # 1. Обновить remaining_work для ВСЕХ активных задач
        resource_consumers = defaultdict(set)
        for active_tool in self.active_tools:
            for r in ResourceType:
                if active_tool.remaining_work[r] > 1e-9:
                    resource_consumers[r].add(active_tool.tool_id)
        
        for active_tool in self.active_tools:
            for r in ResourceType:
                if active_tool.remaining_work[r] > 1e-9:
                    # Вычислить fair share для этого ресурса
                    num_consumers = len(resource_consumers[r])
                    capacity = self.resource_manager.get_capacity(r)
                    share = capacity / num_consumers
                    
                    # Выполненная работа за time_delta
                    work_done = share * time_delta
                    
                    # Обновить remaining_work
                    active_tool.remaining_work[r] = max(0, active_tool.remaining_work[r] - work_done)
        
        # 2. Проверить, завершилась ли задача tool полностью
        if all(tool.remaining_work[r] <= 1e-9 for r in ResourceType):
            tool.status = ToolStatus.COMPLETED
            tool.finish_time = self.current_time
            self.active_tools.remove(tool)
            
            # 3. Проверить зависимости и запустить dependent tasks
            self._check_and_start_dependents(tool)
    
    def _check_and_start_dependents(self, finished_tool: ToolInstance) -> None:
        """
        Проверяет, можно ли запустить задачи, зависящие от finished_tool,
        и добавляет события TOOL_START в EventQueue.
        """
        request = self.requests[finished_tool.request_id]
        dag = request.dag
        
        # Найти всех dependents в DAG
        for dependent_name in dag.graph.successors(finished_tool.tool_name):
            dependent_tool = request.tool_instances[dependent_name]
            
            # Проверить, все ли dependencies завершены
            all_deps_done = all(
                request.tool_instances[dep_name].status == ToolStatus.COMPLETED
                for dep_name in dag.graph.predecessors(dependent_name)
            )
            
            if all_deps_done and dependent_tool.status == ToolStatus.PENDING:
                # Добавить событие TOOL_START в очередь
                event = Event(
                    event_type=EventType.TOOL_START,
                    timestamp=self.current_time,
                    payload={'tool_id': dependent_tool.tool_id}
                )
                self.schedule_event(event)
        
        # Проверить, завершён ли весь запрос
        if all(t.status == ToolStatus.COMPLETED for t in request.tool_instances.values()):
            request.finish_time = self.current_time
            # Записать метрику latency
            latency = request.finish_time - request.arrival_time
            self.metrics.record_request_latency(request.request_type, latency)
```

---

## 5. РАСЧЁТ МЕТРИК

### 5.1 Latency метрики

Для каждого запроса:
```
Latency = finish_time - arrival_time
```

Агрегированные метрики по типу запроса:
- **Mean latency**: `E[Latency]`
- **Median (p50)**: 50-й перцентиль
- **p95, p99**: 95-й и 99-й перцентили (для SLA)
- **Max latency**: worst-case

### 5.2 Throughput метрики

```
Throughput = total_completed_requests / simulation_duration  (запросов/сек)

или в запросах/мин:
Throughput_per_min = Throughput × 60
```

### 5.3 Resource Utilization

Для каждого ресурса `R` в момент времени `t`:
```
Utilization[R, t] = (∑ allocated[i, R]) / total_capacity[R]

где allocated[i, R] = share инструмента i на ресурсе R
```

**Временное усреднение:**
```
Mean_Utilization[R] = (1/T) ∫₀ᵀ Utilization[R, t] dt

Численная аппроксимация (trapezoidal rule):
Mean_Utilization[R] ≈ ∑ᵢ (Uᵢ + Uᵢ₊₁)/2 × (tᵢ₊₁ - tᵢ) / T
```

### 5.4 Queue depth (опциональная метрика)

```
Queue_depth[t] = количество инструментов со статусом PENDING или RUNNING
```

Помогает понять насыщение системы.

---

## 6. СЦЕНАРИИ ЭКСПЕРИМЕНТОВ

### 6.1 Прямая задача: Latency при заданном Throughput

**Формулировка:**
> При arrival rate λ = 100 запросов/мин типа `web-search`, какая средняя latency?

**Конфигурация эксперимента:**
```yaml
# configs/scenarios/web_search_load_test.yaml

scenario_name: "WebSearch Load Test"
description: "Estimate latency for web-search requests at 100 req/min"

simulation:
  duration: 3600  # 1 час симуляции (секунды)
  random_seed: 42
  num_runs: 10    # для статистической значимости

workload:
  request_types:
    - type: "web-search"
      dag_template: "WebSearchDAG"  # ссылка на конфиг агентов
      arrival_rate: 100  # запросов/мин
      arrival_distribution: "poisson"

resources:
  cpu:
    total_capacity: 1000  # условные единицы (например, CPU-seconds/sec = количество ядер)
  npu:
    total_capacity: 256
  memory:
    total_capacity: 1024000  # MB
  network:
    total_capacity: 10000  # MB/s
  disk:
    total_capacity: 5000  # MB/s

metrics:
  - latency_mean
  - latency_p50
  - latency_p95
  - latency_p99
  - throughput
  - resource_utilization
```

**Алгоритм запуска:**
```python
def run_scenario(scenario_config: Dict) -> ExperimentResult:
    results = []
    
    for run_id in range(scenario_config['simulation']['num_runs']):
        # Создать engine с заданным random_seed
        seed = scenario_config['simulation']['random_seed'] + run_id
        np.random.seed(seed)
        
        engine = SimulationEngine(...)
        
        # Сгенерировать arrival events
        arrival_times = generate_arrival_times(
            lambda_per_min=scenario_config['workload']['request_types'][0]['arrival_rate'],
            simulation_duration=scenario_config['simulation']['duration']
        )
        
        for t in arrival_times:
            event = Event(
                timestamp=t,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={'request_type': 'web-search'}
            )
            engine.schedule_event(event)
        
        # Запустить симуляцию
        engine.run(until=scenario_config['simulation']['duration'])
        
        # Собрать метрики
        metrics = engine.metrics.compute_final_metrics()
        results.append(metrics)
    
    # Агрегировать по runs
    return aggregate_results(results)
```

### 6.2 Обратная задача: Throughput при SLA по Latency

**Формулировка:**
> Какой максимальный throughput возможен для `deep-research` запросов, если SLA требует p95_latency < 30 секунд?

**Подход: Binary Search по throughput**

```python
def find_max_throughput_with_sla(
    request_type: str,
    sla_metric: str,  # "latency_p95"
    sla_threshold: float,  # 30 секунд
    tolerance: float = 0.1  # точность поиска
) -> float:
    """
    Бинарный поиск максимального throughput при соблюдении SLA.
    
    Returns:
        Максимальный throughput (запросов/мин)
    """
    
    # Начальные границы поиска
    low_throughput = 1.0   # запросов/мин
    high_throughput = 1000.0
    
    best_throughput = 0.0
    
    while high_throughput - low_throughput > tolerance:
        mid_throughput = (low_throughput + high_throughput) / 2.0
        
        # Запустить симуляцию с mid_throughput
        scenario = create_scenario(request_type, mid_throughput)
        result = run_scenario(scenario)
        
        observed_metric = result.metrics[sla_metric]
        
        if observed_metric <= sla_threshold:
            # SLA соблюдается, можно увеличить throughput
            best_throughput = mid_throughput
            low_throughput = mid_throughput
        else:
            # SLA нарушается, нужно снизить throughput
            high_throughput = mid_throughput
    
    return best_throughput
```

**Оптимизация:** Можно использовать результаты предыдущих runs для warm-start (если latency монотонна по throughput).

### 6.3 Mixed Workload (множественные типы запросов)

**Формулировка:**
> Смесь запросов: 60% web-search (λ₁=60/мин), 30% product-matching (λ₂=30/мин), 10% deep-research (λ₃=10/мин). Какие latency для каждого типа?

**Конфигурация:**
```yaml
workload:
  request_types:
    - type: "web-search"
      dag_template: "WebSearchDAG"
      arrival_rate: 60
      arrival_distribution: "poisson"
    
    - type: "product-matching"
      dag_template: "ProductMatchingDAG"
      arrival_rate: 30
      arrival_distribution: "poisson"
    
    - type: "deep-research"
      dag_template: "DeepResearchDAG"
      arrival_rate: 10
      arrival_distribution: "poisson"
```

**Генерация arrivals:** Merge нескольких Poisson процессов (superposition property).

---

## 7. ВАЛИДАЦИЯ И ТЕСТИРОВАНИЕ

### 7.1 Unit тесты

1. **EventQueue correctness:**
   - События извлекаются в порядке возрастания timestamp
   - События с одинаковым timestamp — в порядке priority
   - Только события START хранятся в очереди

2. **Find next completion logic:**
   - Один инструмент на одном ресурсе → получает 100% capacity
   - Два идентичных инструмента на одном ресурсе → по 50% каждый
   - Корректное определение ближайшего завершения среди N задач и M ресурсов

3. **Remaining work updates:**
   - После time_delta все активные задачи обновляют remaining_work корректно
   - Fair-share распределение соблюдается (каждая задача получает capacity/N)

4. **DAG dependency resolution:**
   - Tool не стартует, пока все dependencies не завершены
   - Завершение всех инструментов в DAG → запрос completed

### 7.2 Integration тесты

1. **Single request, single tool:**
   - Latency = work_needed / resource_capacity
   - Проверка формулы без resource contention

2. **Two sequential tools (A → B):**
   - Latency = time(A) + time(B)
   - Tool B стартует сразу после завершения A

3. **Two parallel tools (A → C ← B):**
   - Tool C стартует после max(finish(A), finish(B))

### 7.3 Стресс-тесты

1. **High load (система насыщена):**
   - Arrival rate → ∞: проверить, что latency растёт, но система не падает
   - Проверить корректность resource sharing при > 1000 active tools

2. **Deep DAG (длинная цепочка зависимостей):**
   - DAG из 50 последовательных инструментов
   - Проверить, что events генерируются корректно

### 7.4 Сравнение с аналитической моделью (санity check)

**Простой случай:** M/M/1 очередь

Если:
- Один инструмент (без DAG)
- Poisson arrivals (λ)
- Exponential service time (μ)

То аналитически:
```
E[Latency] = 1 / (μ - λ)  (при λ < μ)
```

Запустить симуляцию и сравнить с формулой (расхождение < 5%).

---

## 8. ОПТИМИЗАЦИИ И TRADE-OFFS

### 8.1 Производительность симулятора

| Компонент | Базовая реализация | Оптимизация | Trade-off |
|-----------|---------------------|-------------|-----------|
| Event Queue | Python heapq (только старты) | heapq (достаточно) | heapq — O(log n), для > 1M событий можно calendar queue |
| Active Tools tracking | Set | Set (достаточно) | O(1) add/remove, memory overhead незначительный |
| Find next completion | O(N_tools × N_resources) | Кеширование resource_consumers | Для > 1000 одновременно активных задач |
| Resource allocation | Per-tool calculation | Vectorized (NumPy) | Для > 10K одновременно активных инструментов |
| Remaining work update | Полный пересчёт всех задач | Только задачи с affected resources | Сложность vs 2x speedup |

**Рекомендация:** Начать с простой реализации (heapq + set + naive loops). Оптимизировать только если профайлинг покажет bottleneck.

**Ключевое преимущество текущего подхода:** 
- EventQueue содержит только старты (обычно << количества завершений)
- Не нужно удалять/перепланировать события завершений при изменении resource sharing
- Явный контроль времени на каждом шаге

### 8.2 Точность vs Speed

**Наш подход обеспечивает 100% точность**, так как:
1. На каждом шаге вычисляем точное время следующего завершения с учётом текущего fair-share
2. Обновляем `remaining_work` для всех активных задач пропорционально прошедшему времени
3. Никакой аппроксимации или дискретизации времени

**Возможная оптимизация (с потерей точности):**
- **Time quantum discretization:** Продвигать время шагами по 1ms вместо точных моментов
- Плюсы: Упрощение логики, меньше вычислений на шаг
- Минусы: Потеря точности до 1ms × N_steps, может быть значительно

**Рекомендация:** Использовать точную симуляцию (continuous time). Дискретизация оправдана только для очень больших систем (> 10K одновременно активных задач).

### 8.3 Memory footprint

При длительных симуляциях (многие часы) хранение всех completed requests может занимать GB.

**Решение:**
- Streaming metrics: вычислять статистики онлайн (running mean, quantiles via t-digest)
- Удалять completed requests после записи метрик

```python
class StreamingMetricsCollector:
    def __init__(self):
        self.latency_digest = TDigest()  # для percentiles
        self.latency_sum = 0.0
        self.latency_count = 0
    
    def record_request_latency(self, request_type: str, latency: float):
        self.latency_digest.update(latency)
        self.latency_sum += latency
        self.latency_count += 1
        
        # НЕ сохраняем request object!
    
    def get_percentile(self, p: float) -> float:
        return self.latency_digest.percentile(p)
```

---

## 9. ПЛАН РЕАЛИЗАЦИИ (ПОЭТАПНЫЙ)

### Phase 1: Базовая инфраструктура (1-2 дня)
**Цель:** Запустить простейшую симуляцию без resource sharing.

Задачи:
1. ✅ Создать классы:
   - `Event`, `EventType`, `EventQueue` (heapq wrapper)
   - `ToolInstance` (экземпляр инструмента для запроса)
   - `Request` (запрос с DAG)

2. ✅ Реализовать `SimulationEngine`:
   - Базовый event loop (без resource manager)
   - Обработка REQUEST_ARRIVAL, TOOL_START, TOOL_FINISH
   - Dependency resolution (проверка DAG)

3. ✅ Unit тесты:
   - Порядок событий в очереди
   - DAG dependency logic (A → B → C)

**Критерий готовности:** Запуск симуляции с одним запросом, состоящим из 3 последовательных инструментов, корректно вычисляет latency.

---

### Phase 2: Resource Management (2-3 дня)
**Цель:** Добавить fair-share распределение ресурсов и логику определения следующего завершения.

Задачи:
1. ✅ Создать `Resource`, `ResourceManager`:
   - Tracking capacity для каждого ресурса
   - Методы `get_capacity(resource_type)`

2. ✅ Реализовать в `SimulationEngine`:
   - `_find_next_completion()` — находит ближайшее завершение среди активных задач
   - `_handle_resource_completion()` — обновляет remaining_work для всех задач
   - Fair-share распределение (capacity / num_consumers)

3. ✅ Добавить `remaining_work` в `ToolInstance`:
   - Dict[ResourceType, float] — оставшаяся работа по каждому ресурсу
   - Инициализация при старте задачи

4. ✅ Unit тесты:
   - 2 инструмента на одном ресурсе → по 50% capacity
   - Корректное определение ближайшего завершения
   - Обновление remaining_work пропорционально time_delta

**Критерий готовности:** Два одновременно запущенных инструмента корректно делят ресурсы 50/50, симуляция продвигается в точные моменты завершения.

---

### Phase 3: Request Generation & Metrics (2 дня)
**Цель:** Генерация множественных запросов, сбор метрик.

Задачи:
1. ✅ `RequestGenerator`:
   - `generate_poisson_arrivals(lambda, duration)`
   - Поддержка множественных типов запросов (mixed workload)

2. ✅ `MetricsCollector`:
   - Запись latency для каждого запроса
   - Streaming computation (mean, p50, p95, p99)
   - Resource utilization tracking

3. ✅ Integration в `SimulationEngine`

**Критерий готовности:** Симуляция 100 запросов типа `web-search` при λ=60/мин, корректный вывод mean latency и throughput.

---

### Phase 4: Эксперименты и Валидация (3-4 дня)
**Цель:** Запуск реальных сценариев, валидация результатов.

Задачи:
1. ✅ Создать YAML конфиги для сценариев:
   - `web_search_load_test.yaml`
   - `deep_research_sla.yaml`
   - `mixed_workload.yaml`

2. ✅ `ExperimentRunner`:
   - Парсинг YAML сценариев
   - Multiple runs с разными seeds
   - Агрегация результатов (mean ± std)

3. ✅ Валидация:
   - Сравнение с аналитической моделью (M/M/1)
   - Стресс-тесты (high load, deep DAG)

4. ✅ Визуализация результатов:
   - Latency vs Throughput графики
   - Resource utilization heatmaps

**Критерий готовности:** Успешное прохождение всех валидационных тестов, получение осмысленных результатов для 3+ сценариев.

---

### Phase 5: Обратная задача (Binary Search) (1-2 дня)
**Цель:** Автоматический поиск максимального throughput при SLA.

Задачи:
1. ✅ Реализовать `find_max_throughput_with_sla()`
2. ✅ Интеграция с `ExperimentRunner`
3. ✅ Тесты: проверить, что найденный throughput действительно соблюдает SLA

**Критерий готовности:** Для сценария `deep-research` с SLA p95 < 30s, система находит max throughput за < 10 итераций binary search.

---

### Phase 6: Оптимизации и Production-Ready (опционально, 2-3 дня)
**Цель:** Повысить производительность и надёжность.

Задачи:
1. ⚡ Кеширование resource_consumers между шагами (если набор активных задач не изменился)
2. ⚡ Streaming metrics (удаление completed requests после записи)
3. ⚡ Профилирование и оптимизация bottlenecks (cProfile, line_profiler)
4. ⚡ Vectorized операции через NumPy для update remaining_work (при > 1K активных задач)
5. 📊 Логирование и debugging tools (детальная трассировка событий)
6. 📊 Export результатов в JSON/CSV для дальнейшего анализа

---

## 10. ОТКРЫТЫЕ ВОПРОСЫ И ДОПУЩЕНИЯ

### 10.1 Допущения

1. **Статическая load на ресурсы:**
   - `tool.get_network_load()` возвращает константу
   - В реальности может зависеть от размера входных данных (которые определяются output предыдущих инструментов)
   - **Решение для v2:** Передавать context между инструментами, динамически вычислять load

2. **Нулевое overhead на переключение:**
   - Предполагаем, что смена активных инструментов (preemption) происходит мгновенно
   - **Решение для v2:** Добавить penalty за context switch (например, 1-10ms)

3. **Идеальный fair-share:**
   - В реальности планировщики имеют granularity (quantum времени)
   - **Решение для v2:** Discrete time simulation с time quantum (например, 10ms)

4. **Независимость ресурсов:**
   - Memory и Network могут иметь сложные взаимодействия (например, memory-mapped I/O)
   - Пока рассматриваем как независимые

### 10.2 Ограничения текущей модели

1. **Нет queueing на ресурсах:**
   - Если инструмент готов стартовать, он стартует немедленно (даже если ресурсы перегружены)
   - Просто получает меньшую долю
   - **Альтернатива:** Вводить admission control (максимум N активных инструментов)

2. **Нет приоритетов:**
   - Все инструменты равноправны
   - **Расширение:** Weighted fair-share (разные веса для разных типов запросов)

3. **Нет failures:**
   - Все инструменты завершаются успешно
   - **Расширение:** Моделировать timeout, retry, fallback

### 10.3 Вопросы для уточнения

1. **Определение resource capacity:**
   - Как интерпретировать `cpu: 1000`? Это CPU-seconds/sec (= количество ядер)?
   - Нужны ли единицы измерения в конфиге?

2. **Latency из конфига (например, `latency: 100`):**
   - Это latency инструмента при 100% ресурсов или при какой-то baseline загрузке?
   - **Предложение:** Убрать фиксированный latency, вычислять из work/capacity

3. **Какие инструменты используют какие ресурсы?**
   - Сейчас в `tools.py` у некоторых инструментов все `get_*_load()` возвращают 0
   - Нужно заполнить реальные значения (или формулы для вычисления)

---

## 11. ВЫВОДЫ И РЕКОМЕНДАЦИИ

### 11.1 Рекомендуемый подход

1. **Event-driven simulation с fair-share:** Оптимальный баланс точности и производительности для данной задачи.

2. **Только события START в очереди:** Ключевое архитектурное решение, которое:
   - Упрощает логику (не нужно удалять/перепланировать FINISH события)
   - Обеспечивает 100% точность (всегда используем актуальное состояние системы)
   - Снижает количество событий в очереди (обычно старты << завершения)

3. **Динамическое вычисление next completion:** На каждом шаге вычисляем момент следующего завершения с учётом текущего fair-share распределения.

4. **Poisson arrivals:** Стандартная модель для моделирования запросов, обеспечивает статистическую обоснованность.

5. **Binary search для обратной задачи:** Эффективный способ найти max throughput при SLA.

### 11.2 Критерии успеха

✅ **Phase 1-3:** Базовая симуляция работает, проходит unit/integration тесты  
✅ **Phase 4:** Результаты валидируются против аналитической модели (< 5% error для M/M/1)  
✅ **Phase 5:** Binary search находит max throughput за < 10 итераций  
✅ **Phase 6 (optional):** Производительность > 100K событий/сек на обычном laptop

### 11.3 Риски и митигация

| Риск | Вероятность | Воздействие | Митигация |
|------|-------------|-------------|-----------|
| Некорректное resource sharing при высокой загрузке | Средняя | Высокое | Extensive unit tests + stress tests |
| Численная нестабильность (floating point) | Низкая | Среднее | Использовать `Decimal` для критических расчётов |
| Медленная симуляция (> 10 мин для одного сценария) | Средняя | Среднее | Профилирование, incremental update, parallel runs |
| Неполные данные в конфигах (missing resource loads) | Высокая | Низкое | Явная валидация конфигов, raise exceptions |

---

## 12. ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ

### 12.1 Литература

1. **Discrete Event Simulation:**
   - Law & Kelton, "Simulation Modeling and Analysis"
   - Banks et al., "Discrete-Event System Simulation"

2. **Queueing Theory:**
   - Kleinrock, "Queueing Systems: Theory"
   - Bertsekas & Gallager, "Data Networks"

3. **Resource Scheduling:**
   - Pinedo, "Scheduling: Theory, Algorithms, and Systems"
   - Dominant Resource Fairness (DRF): Ghodsi et al., NSDI 2011

### 12.2 Формулы для Quick Reference

**M/M/1 Queue (аналитическая модель):**
```
λ = arrival rate
μ = service rate (capacity)
ρ = λ/μ (utilization)

E[Latency] = 1/(μ - λ)  [для λ < μ]
E[Queue Length] = ρ/(1 - ρ)
```

**Little's Law:**
```
L = λ × W

где:
L = среднее количество запросов в системе
λ = arrival rate
W = среднее время в системе (latency)
```

**Fair Share:**
```
Share[i, r] = Capacity[r] / |ActiveTools[r]|

Completion_time[i] = max{ Remaining_work[i, r] / Share[i, r] : r ∈ Resources[i] }
```

---

## КОНЕЦ ДОКУМЕНТА

**Следующий шаг:** Утверждение плана и переход к Phase 1 реализации.

**Вопросы для обсуждения:**
1. Согласны ли с выбором Poisson процесса для arrivals?
2. Нужны ли приоритеты для разных типов запросов?
3. Какие реальные значения resource capacities использовать?
4. Достаточно ли 5 типов ресурсов (CPU/NPU/Memory/Network/Disk) или нужно больше granularity?

