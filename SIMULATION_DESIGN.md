# Архитектура дискретно-событийной симуляции мультиагентной системы

**Автор:** Principal Systems Architect & Simulation Engineer  
**Дата:** 12 октября 2025  
**Статус:** Design & Planning Phase

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

| Тип события | Описание | Payload | Триггеры |
|-------------|----------|---------|----------|
| `REQUEST_ARRIVAL` | Поступление нового запроса в систему | `{request_id, request_type}` | Генерируется заранее по Poisson процессу |
| `TOOL_START` | Начало выполнения инструмента | `{tool_id, request_id, tool_name}` | Завершение всех dependencies OR REQUEST_ARRIVAL для root tools |
| `TOOL_FINISH` | Завершение выполнения инструмента | `{tool_id, request_id, tool_name}` | Истечение времени выполнения с учётом resource sharing |

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

**Формула времени завершения (naive approach):**

Для инструмента `i`, использующего ресурсы `{r₁, r₂, ..., rₖ}`:

```
Work_needed[i, r] = load[i, r]  (total work для ресурса r)

Share[i, r, t] = C_r / N_r(t)   (где N_r(t) = количество активных инструментов, использующих r в момент t)

Completion_time[i, r] = remaining_work[i, r] / Share[i, r]

Actual_completion_time[i] = max{ Completion_time[i, r] : r ∈ resources[i] }
(лимитирующий ресурс - bottleneck)
```

**Проблема:** При изменении `N_r(t)` (старт/завершение других инструментов) необходим пересчёт!

### 3.3 Алгоритм пересчёта времён завершения (Resource Sharing)

**Вариант 1: Пересчёт на каждое событие** ✓ (рекомендуется для точности)

```python
def recalculate_finish_times(current_time: float, active_tools: Set[ToolInstance]) -> Dict[str, float]:
    """
    Пересчитывает времена завершения всех активных инструментов
    с учётом текущего fair-share распределения.
    
    Returns:
        Dict[tool_id -> predicted_finish_time]
    """
    # 1. Подсчитать количество активных инструментов для каждого ресурса
    resource_consumers = defaultdict(set)
    for tool in active_tools:
        for resource_type in ResourceType:
            load = tool.get_load(resource_type)
            if load > 0:
                resource_consumers[resource_type].add(tool.tool_id)
    
    # 2. Для каждого инструмента найти bottleneck resource
    finish_times = {}
    for tool in active_tools:
        max_time = current_time
        
        for resource_type in ResourceType:
            load = tool.get_load(resource_type)
            if load <= 0:
                continue
            
            # Fair share для этого ресурса
            num_consumers = len(resource_consumers[resource_type])
            share = resources[resource_type].total_capacity / num_consumers
            
            # Оставшаяся работа
            remaining = tool.remaining_work[resource_type]
            
            # Время до завершения по этому ресурсу
            time_to_finish = current_time + (remaining / share)
            max_time = max(max_time, time_to_finish)
        
        finish_times[tool.tool_id] = max_time
    
    return finish_times
```

**Сложность:** O(|ActiveTools| × |ResourceTypes|) на каждое событие  
**Плюсы:** Точный учёт динамического перераспределения  
**Минусы:** Может быть дорого при большом количестве одновременно активных инструментов

**Вариант 2: Инкрементальный update (оптимизация)**

Пересчитывать только инструменты, использующие те же ресурсы, что и завершённый/стартовавший.

```python
def incremental_update(event: Event, active_tools: Set[ToolInstance], 
                       affected_resources: Set[ResourceType]) -> None:
    """
    Обновляет только инструменты, затронутые изменением в affected_resources.
    """
    # Найти инструменты, использующие affected resources
    affected_tools = {
        tool for tool in active_tools 
        if any(tool.get_load(r) > 0 for r in affected_resources)
    }
    
    # Пересчитать finish_times только для них
    # ...
```

**Критерий выбора:** Если средняя загрузка системы < 50%, используем Вариант 1 (простота важнее). При > 70% загрузке — Вариант 2.

### 3.4 Обработка зависимостей (DAG Dependency Resolution)

При завершении инструмента `T_finished`:

```python
def handle_tool_finish(tool_id: str, current_time: float) -> List[Event]:
    """
    Обрабатывает завершение инструмента и генерирует TOOL_START события
    для готовых к запуску зависимых инструментов.
    
    Returns:
        Список новых событий TOOL_START
    """
    finished_tool = tool_instances[tool_id]
    finished_tool.status = ToolStatus.COMPLETED
    finished_tool.finish_time = current_time
    
    request = requests[finished_tool.request_id]
    dag = request.dag
    
    new_events = []
    
    # Найти всех dependents в DAG
    for dependent_name in dag.graph.successors(finished_tool.tool_name):
        dependent_tool = request.tool_instances[dependent_name]
        
        # Проверить, все ли dependencies завершены
        all_deps_done = all(
            request.tool_instances[dep_name].status == ToolStatus.COMPLETED
            for dep_name in dag.graph.predecessors(dependent_name)
        )
        
        if all_deps_done and dependent_tool.status == ToolStatus.PENDING:
            # Создать событие TOOL_START
            event = Event(
                event_type=EventType.TOOL_START,
                timestamp=current_time,  # старт немедленно
                payload={'tool_id': dependent_tool.tool_id}
            )
            new_events.append(event)
    
    # Проверить, завершён ли весь запрос
    if all(t.status == ToolStatus.COMPLETED for t in request.tool_instances.values()):
        request.finish_time = current_time
        # Записать метрику latency
        latency = request.finish_time - request.arrival_time
        metrics.record_request_latency(request.request_type, latency)
    
    return new_events
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
│   ├── scheduler.py               # [NEW] Fair-share scheduler
│   └── simulation_engine.py       # [NEW] Основной цикл симуляции
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
from typing import List, Dict, Set
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    REQUEST_ARRIVAL = 1
    TOOL_START = 2
    TOOL_FINISH = 3

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
    """
    
    def __init__(self, 
                 resource_manager: ResourceManager,
                 scheduler: FairShareScheduler,
                 metrics_collector: MetricsCollector):
        self.event_queue: List[Event] = []  # heapq
        self.current_time: float = 0.0
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.metrics = metrics_collector
        
        self.requests: Dict[UUID, Request] = {}
        self.active_tools: Dict[str, ToolInstance] = {}
        
    def schedule_event(self, event: Event) -> None:
        """Добавляет событие в очередь."""
        heappush(self.event_queue, event)
    
    def run(self, until: float) -> None:
        """
        Основной цикл симуляции.
        
        Args:
            until: время окончания симуляции (секунды)
        """
        while self.event_queue and self.event_queue[0].timestamp <= until:
            event = heappop(self.event_queue)
            self.current_time = event.timestamp
            
            # Dispatch по типу события
            if event.event_type == EventType.REQUEST_ARRIVAL:
                self._handle_request_arrival(event)
            elif event.event_type == EventType.TOOL_START:
                self._handle_tool_start(event)
            elif event.event_type == EventType.TOOL_FINISH:
                self._handle_tool_finish(event)
            
            # После каждого события обновляем метрики
            self.metrics.snapshot(self.current_time, 
                                  self.active_tools, 
                                  self.resource_manager)
    
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
        tool = self.active_tools[tool_id]
        tool.status = ToolStatus.RUNNING
        tool.start_time = self.current_time
        
        # Allocate resources
        self.resource_manager.allocate(tool)
        
        # Пересчитать finish times для всех active tools
        finish_times = self.scheduler.recalculate_finish_times(
            self.current_time, self.active_tools.values()
        )
        
        # Обновить/создать события TOOL_FINISH
        for tid, finish_time in finish_times.items():
            # Удалить старое событие TOOL_FINISH (если было)
            self._remove_tool_finish_event(tid)
            
            # Добавить новое
            finish_event = Event(
                timestamp=finish_time,
                event_type=EventType.TOOL_FINISH,
                payload={'tool_id': tid}
            )
            self.schedule_event(finish_event)
    
    def _handle_tool_finish(self, event: Event) -> None:
        """Обработка завершения выполнения инструмента."""
        tool_id = event.payload['tool_id']
        tool = self.active_tools[tool_id]
        tool.status = ToolStatus.COMPLETED
        tool.finish_time = self.current_time
        
        # Release resources
        self.resource_manager.release(tool)
        
        # Удалить из active
        del self.active_tools[tool_id]
        
        # Найти и запустить dependent tools (см. handle_tool_finish из раздела 3.4)
        new_start_events = self._check_and_start_dependents(tool)
        for evt in new_start_events:
            self.schedule_event(evt)
        
        # Пересчитать finish times для оставшихся active tools
        if self.active_tools:
            finish_times = self.scheduler.recalculate_finish_times(
                self.current_time, self.active_tools.values()
            )
            
            # Обновить события
            for tid, finish_time in finish_times.items():
                self._remove_tool_finish_event(tid)
                finish_event = Event(
                    timestamp=finish_time,
                    event_type=EventType.TOOL_FINISH,
                    payload={'tool_id': tid}
                )
                self.schedule_event(finish_event)
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

2. **FairShareScheduler:**
   - Один инструмент на одном ресурсе → получает 100% capacity
   - Два идентичных инструмента → по 50% каждый
   - Bottleneck resource определяет finish time

3. **DAG dependency resolution:**
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
| Event Queue | Python heapq | heapq (достаточно) | heapq — O(log n), для > 1M событий можно calendar queue |
| Active Tools tracking | Dict | Dict (достаточно) | O(1) lookup, memory overhead незначительный |
| Recalculate finish times | Full recalc каждый event | Incremental update | Сложность реализации vs 2-3x speedup при высокой загрузке |
| Resource allocation | Per-tool calculation | Vectorized (NumPy) | Для > 10K одновременно активных инструментов |

**Рекомендация:** Начать с простой реализации (heapq + dict + full recalc). Оптимизировать только если профайлинг покажет bottleneck.

### 8.2 Точность vs Speed

**Вариант 1: Точная симуляция**
- Пересчёт finish times на каждое событие
- Учёт всех зависимостей в DAG
- Плюсы: максимальная точность
- Минусы: медленно для длинных симуляций (> 1 млн событий)

**Вариант 2: Аппроксимация**
- Пересчёт только при "значительных" изменениях (например, изменение загрузки > 10%)
- Плюсы: 5-10x быстрее
- Минусы: потеря точности до 5-10% в latency

**Критерий выбора:** Для design space exploration (множество сценариев) — Вариант 2. Для финальной валидации — Вариант 1.

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
**Цель:** Добавить fair-share распределение ресурсов.

Задачи:
1. ✅ Создать `Resource`, `ResourceManager`:
   - Tracking активных consumers для каждого ресурса
   - Методы `allocate()`, `release()`

2. ✅ Реализовать `FairShareScheduler`:
   - `recalculate_finish_times()` с учётом fair share
   - Определение bottleneck resource

3. ✅ Интегрировать в `SimulationEngine`:
   - При TOOL_START: allocate + recalc
   - При TOOL_FINISH: release + recalc

4. ✅ Unit тесты:
   - 2 инструмента на одном ресурсе → по 50% capacity
   - Bottleneck resource определяет finish time

**Критерий готовности:** Два одновременно запущенных инструмента корректно делят ресурсы 50/50, и их finish times вычисляются правильно.

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
1. ⚡ Incremental update для finish times (вместо full recalc)
2. ⚡ Streaming metrics (удаление completed requests)
3. ⚡ Профилирование и оптимизация bottlenecks
4. 📊 Логирование и debugging tools
5. 📊 Export результатов в JSON/CSV для дальнейшего анализа

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

2. **Poisson arrivals:** Стандартная модель для моделирования запросов, обеспечивает статистическую обоснованность.

3. **Полный пересчёт finish times:** Простота реализации и отладки важнее для первой версии.

4. **Binary search для обратной задачи:** Эффективный способ найти max throughput при SLA.

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

