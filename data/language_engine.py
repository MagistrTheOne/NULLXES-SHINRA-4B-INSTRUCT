"""Deterministic, NULLXES-authored prose for tokenizer coverage only.

This corpus is deliberately separate from the verified task records consumed by
the model packer. The generator composes authored propositions into short
articles; it does not sample external text or generate vocabulary filler.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import re
import sqlite3
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Iterator

ENGINE_VERSION = "nullxes-native-language-engine.v2"
SCHEMA = "nullxes.shinra.tokenizer-language-corpus.v2"
RECORD_SCHEMA = "nullxes.shinra.tokenizer-language-record.v2"
LANGUAGES = ("en", "ru")

# Each entry is an authored, bilingual concept with a definition, a causal
# relation, a practical consequence, and a limitation. Keeping these clauses
# attached to one concept prevents arbitrary cross-domain sentence mixing.
CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("feedback_control", "control", "A feedback controller compares a measured output with a target.", "The difference between them is an error signal.", "The controller changes its input to reduce that error.", "A delayed measurement can make a correct adjustment arrive too late.", "Регулятор с обратной связью сравнивает измеренный выход с заданным значением.", "Разность между ними образует сигнал ошибки.", "Регулятор меняет управляющее воздействие, чтобы уменьшить эту ошибку.", "Запаздывающий замер может привести к тому, что верная поправка поступит слишком поздно."),
    ("sensors", "physical_ai", "A sensor converts a physical quantity into a measurable signal.", "Calibration relates that signal to a stated unit and reference.", "A robot can use the calibrated value to estimate its surroundings.", "Noise and occlusion still limit what the sensor can establish.", "Датчик преобразует физическую величину в измеримый сигнал.", "Калибровка связывает сигнал с указанной единицей и эталоном.", "Робот использует откалиброванное значение для оценки окружения.", "Шум и перекрытие обзора всё равно ограничивают выводы по данным датчика."),
    ("state_estimation", "physical_ai", "A state estimate combines observations with a model of how a system changes.", "The estimate is updated when a new observation arrives.", "A controller can then choose an action using the current estimate.", "Uncertainty must remain explicit when observations are incomplete.", "Оценка состояния объединяет наблюдения с моделью изменения системы.", "При поступлении нового наблюдения оценка обновляется.", "Затем регулятор выбирает действие с учётом текущей оценки.", "Если наблюдения неполны, неопределённость следует сохранять явно."),
    ("collision_avoidance", "physical_ai", "Collision avoidance compares a planned path with observed obstacles.", "A safety margin accounts for sensing and braking uncertainty.", "The system slows down or stops when the remaining distance is insufficient.", "A clear sensor view is not proof that every part of the route is safe.", "Обход столкновений сопоставляет планируемый путь с обнаруженными препятствиями.", "Запас безопасности учитывает погрешность измерения и торможения.", "Система замедляется или останавливается, если расстояния недостаточно.", "Свободный обзор датчика не доказывает безопасность всего маршрута."),
    ("planning", "reasoning", "A plan is an ordered set of actions that is intended to reach a goal.", "Each action has preconditions and an expected effect.", "The planner checks whether an action is valid in the current state.", "A changed state can invalidate the remaining plan.", "План — это упорядоченный набор действий для достижения цели.", "У каждого действия есть предварительные условия и ожидаемый эффект.", "Планировщик проверяет допустимость действия в текущем состоянии.", "Изменившееся состояние может сделать оставшуюся часть плана недействительной."),
    ("cancellation", "instruction_following", "A cancellation request changes the status of a pending action.", "The system checks whether execution has already begun.", "If the action is still pending, it is removed from the queue.", "A completed action cannot be undone by changing its queue entry.", "Запрос отмены меняет статус ожидающего действия.", "Система проверяет, началось ли уже выполнение.", "Если действие ещё ожидает запуска, его удаляют из очереди.", "Завершённое действие нельзя отменить простым изменением записи в очереди."),
    ("algorithms", "computing", "An algorithm defines a finite sequence of steps for a specified problem.", "A precondition describes the inputs for which those steps are valid.", "An invariant helps verify that intermediate states preserve correctness.", "Termination and correctness are separate properties.", "Алгоритм задаёт конечную последовательность шагов для определённой задачи.", "Предусловие описывает входные данные, для которых шаги допустимы.", "Инвариант помогает проверить корректность промежуточных состояний.", "Завершение и правильность — разные свойства алгоритма."),
    ("complexity", "computing", "Complexity describes how resource use changes with input size.", "An asymptotic bound compares growth rates rather than one measured run.", "The bound helps identify workloads that may become expensive.", "It does not replace measurements on the target hardware.", "Сложность описывает, как расход ресурсов меняется с размером входа.", "Асимптотическая оценка сравнивает скорости роста, а не один замер.", "Она помогает выявить задачи, которые могут стать дорогими.", "Такая оценка не заменяет измерения на целевом оборудовании."),
    ("memory", "computing", "A program's memory use includes persistent state and temporary working buffers.", "Peak use can exceed the size of the final output.", "Chunking reduces some temporary buffers when computation is structured accordingly.", "The real peak still depends on autograd and kernel lifetimes.", "Расход памяти программы включает постоянное состояние и временные буферы.", "Пиковый расход может превышать размер итогового результата.", "Разбиение на блоки уменьшает часть временных буферов при подходящей организации вычислений.", "Фактический пик также зависит от autograd и времени жизни буферов ядра."),
    ("numerical_precision", "computing", "Finite-precision arithmetic rounds values to a representable set.", "Repeated operations can accumulate rounding error.", "A stable algorithm limits error growth for its intended range.", "Agreement on one input does not prove stability on every input.", "В арифметике конечной точности значения округляются до представимого набора.", "При повторных операциях ошибка округления может накапливаться.", "Устойчивый алгоритм ограничивает рост ошибки в заданном диапазоне.", "Совпадение на одном входе не доказывает устойчивость на всех входах."),
    ("causality", "reasoning", "A causal explanation identifies a change that contributes to an outcome.", "A correlation alone does not establish that relationship.", "An intervention can test whether changing one factor changes the result.", "The conclusion depends on which other conditions were held fixed.", "Причинное объяснение указывает изменение, которое влияет на результат.", "Одна корреляция сама по себе не устанавливает такую связь.", "Вмешательство проверяет, изменится ли результат при изменении одного фактора.", "Вывод зависит от того, какие остальные условия оставались постоянными."),
    ("state_machines", "computing", "A state machine represents a system with explicit states and transitions.", "A transition is allowed only when its condition is satisfied.", "Rejecting an invalid transition preserves the previous state.", "The transition table must cover every supported command.", "Конечный автомат представляет систему через явные состояния и переходы.", "Переход разрешён только при выполнении его условия.", "Отклонение недопустимого перехода сохраняет предыдущее состояние.", "Таблица переходов должна охватывать все поддерживаемые команды."),
    ("robot_motion", "physical_ai", "A motion command specifies a direction, speed, and reference frame.", "The reference frame determines how coordinates are interpreted.", "A controller translates the command into actuator signals.", "Actuator limits can prevent the requested motion.", "Команда движения задаёт направление, скорость и систему отсчёта.", "Система отсчёта определяет трактовку координат.", "Регулятор преобразует команду в сигналы исполнительным механизмам.", "Ограничения приводов могут не позволить выполнить запрошенное движение."),
    ("language_ambiguity", "language", "A sentence can admit more than one interpretation when a phrase has unclear scope.", "Context narrows the plausible readings but may not remove all ambiguity.", "A precise instruction states the object and the intended operation.", "When the missing detail changes the result, asking a question is safer than guessing.", "Предложение допускает несколько трактовок, если область действия фразы неясна.", "Контекст сужает круг вариантов, но не всегда устраняет неоднозначность.", "Точная инструкция называет объект и требуемое действие.", "Если недостающая деталь меняет результат, безопаснее уточнить её, чем угадывать."),
    ("dialogue_state", "language", "A dialogue state contains the relevant facts established in earlier turns.", "A new instruction can update or cancel part of that state.", "The response should use the latest compatible instruction.", "Old context should not override an explicit correction.", "Состояние диалога содержит важные факты из предыдущих реплик.", "Новая инструкция может обновить или отменить часть этого состояния.", "В ответе следует учитывать последнюю совместимую инструкцию.", "Старый контекст не должен отменять явное исправление."),
    ("materials", "engineering", "A material property describes how a substance responds under stated conditions.", "Temperature and loading can change the measured response.", "Engineers use those measurements to select a suitable component.", "A value without units or test conditions is incomplete evidence.", "Свойство материала описывает реакцию вещества при заданных условиях.", "Температура и нагрузка могут изменить измеренный отклик.", "Инженеры используют такие измерения при выборе компонента.", "Значение без единиц и условий испытания является неполным свидетельством."),
    ("energy", "physics", "Energy accounting tracks transfers between a system and its surroundings.", "A conversion changes the form of energy, not the conservation rule.", "Losses often appear as heat or other uncollected output.", "A practical balance must state its boundary and measurement interval.", "Энергетический баланс учитывает обмен системы с окружением.", "Преобразование меняет форму энергии, но не закон её сохранения.", "Потери часто проявляются как тепло или иной неучтённый выход.", "Практический баланс должен задавать границу системы и интервал измерения."),
    ("statistics", "reasoning", "A sample is a subset used to estimate properties of a larger population.", "The sampling method affects which observations are represented.", "An interval communicates uncertainty around an estimate.", "A narrow interval is not reliable when the sample is biased.", "Выборка — подмножество, по которому оценивают свойства всей совокупности.", "Метод отбора влияет на то, какие наблюдения представлены.", "Интервал показывает неопределённость оценки.", "Узкий интервал ненадёжен, если выборка смещена."),
    ("safety_interlock", "physical_ai", "A safety interlock blocks an operation while a required condition is false.", "The condition is checked before the operation is enabled.", "A failed check keeps the actuator in its safe state.", "The check itself must be monitored for faults.", "Блокировка безопасности запрещает операцию, пока обязательное условие ложно.", "Условие проверяется до разрешения операции.", "При неудачной проверке привод остаётся в безопасном состоянии.", "Саму проверку также необходимо контролировать на отказ."),
    ("version_control", "engineering", "A version identifier binds an artifact to a specific set of source bytes.", "A content hash detects changes to those bytes.", "Reproducibility requires recording inputs as well as the output hash.", "A matching name alone does not prove that two artifacts are identical.", "Идентификатор версии связывает артефакт с конкретными исходными байтами.", "Хэш содержимого обнаруживает изменения этих байтов.", "Для воспроизводимости фиксируют входы вместе с хэшем результата.", "Совпадение имени само по себе не доказывает идентичность артефактов."),
    ("distributed_systems", "computing", "A distributed service coordinates work across machines that communicate through a network.", "Messages can be delayed, duplicated, or lost even when each machine follows its local rules.", "A protocol defines how participants detect failure and recover shared state.", "A timeout is evidence of missing communication, not proof that a remote operation failed.", "Распределённая служба координирует работу машин, обменивающихся сообщениями по сети.", "Сообщения могут задерживаться, дублироваться или теряться, даже если каждая машина соблюдает локальные правила.", "Протокол определяет, как участники обнаруживают сбои и восстанавливают общее состояние.", "Истечение времени ожидания указывает на отсутствие связи, но не доказывает отказ удалённой операции."),
    ("database_transactions", "computing", "A transaction groups changes that must satisfy a stated consistency rule.", "Isolation controls which intermediate changes concurrent operations can observe.", "A commit makes the accepted update durable according to the storage system's contract.", "A retry is safe only when the operation's effects are understood or made idempotent.", "Транзакция объединяет изменения, которые должны соблюдать заданное правило согласованности.", "Изоляция определяет, какие промежуточные изменения видят параллельные операции.", "Фиксация делает принятое обновление устойчивым согласно контракту хранилища.", "Повтор безопасен, только если последствия операции понятны или она идемпотентна."),
    ("concurrency", "computing", "Concurrent tasks can make progress during overlapping intervals.", "A shared mutable value creates a race when updates are not coordinated.", "Locks, ownership, or message passing can establish a safe ordering.", "Adding workers does not guarantee higher throughput when they contend for one resource.", "Параллельные задачи могут продвигаться в течение пересекающихся интервалов.", "Общее изменяемое значение создаёт гонку при несогласованных обновлениях.", "Блокировки, владение данными или обмен сообщениями задают безопасный порядок.", "Дополнительные исполнители не гарантируют рост пропускной способности при конкуренции за один ресурс."),
    ("network_protocols", "computing", "A network protocol specifies the format and order of messages between endpoints.", "Framing lets a receiver distinguish one message from the next.", "Validation rejects data that violates the protocol before it changes application state.", "Encryption protects selected properties only when keys and endpoints are handled correctly.", "Сетевой протокол задаёт формат и порядок сообщений между узлами.", "Форматирование потока позволяет получателю отделять одно сообщение от следующего.", "Проверка отклоняет данные, нарушающие протокол, до изменения состояния приложения.", "Шифрование защищает отдельные свойства только при корректной работе с ключами и узлами."),
    ("software_testing", "engineering", "A test checks an observable claim about a program under specified conditions.", "A deterministic test fixes its inputs so that a failure can be reproduced.", "Boundary cases reveal behavior near limits that ordinary examples may miss.", "Passing tests provide evidence for covered behavior, not a proof about every possible input.", "Тест проверяет наблюдаемое утверждение о программе при заданных условиях.", "Детерминированный тест фиксирует входы, чтобы сбой можно было воспроизвести.", "Граничные случаи выявляют поведение около пределов, незаметное на обычных примерах.", "Успешные тесты подтверждают проверенное поведение, но не доказывают корректность для всех входов."),
    ("scientific_measurement", "physics", "A measurement compares an observation with a defined unit and procedure.", "Calibration estimates how an instrument's readings relate to a reference.", "Uncertainty describes a range of values consistent with the measurement process.", "Reporting more digits does not reduce uncertainty in the underlying observation.", "Измерение сопоставляет наблюдение с определённой единицей и процедурой.", "Калибровка оценивает связь показаний прибора с эталоном.", "Неопределённость описывает диапазон значений, совместимых с процессом измерения.", "Дополнительные цифры в записи не уменьшают неопределённость самого наблюдения."),
    ("thermodynamics", "physics", "A thermodynamic system is defined by the boundary chosen for its analysis.", "Energy can cross that boundary as heat or work.", "A balance accounts for transfers over a stated interval.", "A conclusion changes if the boundary or omitted transfer changes.", "Термодинамическую систему определяют границей, выбранной для анализа.", "Энергия пересекает эту границу в форме тепла или работы.", "Баланс учитывает переносы за указанный интервал.", "Вывод изменится, если изменить границу или учесть пропущенный перенос."),
    ("materials_fatigue", "engineering", "Repeated loading can initiate and extend a crack in a material.", "The risk depends on stress range, geometry, environment, and prior damage.", "Inspection can detect some defects before they cause a component failure.", "A single static strength value does not describe fatigue life by itself.", "Повторная нагрузка может зародить и постепенно увеличить трещину в материале.", "Риск зависит от диапазона напряжений, геометрии, среды и прежних повреждений.", "Осмотр может обнаружить часть дефектов до отказа детали.", "Одно значение статической прочности само по себе не описывает усталостный ресурс."),
    ("robot_localization", "physical_ai", "A mobile robot estimates its pose relative to a chosen map or frame.", "Motion sensors predict how that pose changes between observations.", "Landmarks can correct accumulated drift when their association is reliable.", "Repeated or ambiguous landmarks can make the estimate uncertain rather than more accurate.", "Мобильный робот оценивает своё положение относительно выбранной карты или системы координат.", "Датчики движения прогнозируют изменение положения между наблюдениями.", "Ориентиры корректируют накопленный дрейф при надёжном сопоставлении.", "Повторяющиеся или неоднозначные ориентиры могут повысить неопределённость оценки."),
    ("robot_manipulation", "physical_ai", "A grasp depends on object geometry, contact, and the forces available at the gripper.", "Visual evidence can estimate a candidate pose but may not reveal hidden surfaces.", "A guarded approach checks contact before applying a larger force.", "The robot should stop and reassess when the object moves outside the observed plan.", "Захват зависит от геометрии объекта, контакта и усилия захвата.", "Изображение помогает оценить положение объекта, но может не показывать скрытые поверхности.", "Контролируемое сближение проверяет контакт до увеличения усилия.", "Роботу следует остановиться и пересмотреть план, если объект сместился за пределы наблюдаемого сценария."),
    ("human_robot_communication", "physical_ai", "A robot command should identify the intended object and requested change.", "A confirmation can expose ambiguity before a physical action begins.", "Status messages distinguish a planned action from one that has completed.", "Silence or a missing sensor report must not be presented as successful execution.", "Команда роботу должна указывать целевой объект и требуемое изменение.", "Подтверждение помогает выявить неоднозначность до начала физического действия.", "Сообщение о статусе отделяет запланированное действие от завершённого.", "Молчание или отсутствие отчёта датчика нельзя выдавать за успешное выполнение."),
    ("probability", "reasoning", "A probability assigns a number to an event under a stated model.", "Conditional probability updates that number when evidence is observed.", "The result depends on the assumptions used to define the model.", "A precise probability is misleading when the relevant assumptions are unstated.", "Вероятность приписывает событию численное значение в рамках заданной модели.", "Условная вероятность обновляет это значение после наблюдения свидетельства.", "Результат зависит от предположений, на которых построена модель.", "Точное число вводит в заблуждение, если важные предположения не указаны."),
    ("causal_intervention", "reasoning", "A causal intervention deliberately changes one factor in a system.", "The outcome is compared with a suitable baseline under controlled conditions.", "Confounding variables can make an observed difference hard to interpret.", "An intervention supports a causal claim only when the design addresses plausible alternatives.", "Причинное вмешательство намеренно меняет один фактор системы.", "Результат сравнивают с подходящей исходной линией при контролируемых условиях.", "Смешивающие факторы затрудняют трактовку наблюдаемого различия.", "Вмешательство подтверждает причинное утверждение лишь тогда, когда дизайн учитывает правдоподобные альтернативы."),
    ("algorithmic_invariants", "computing", "An invariant states a property that should remain true across selected algorithm steps.", "Initialization establishes the property before the loop or recursion begins.", "Each transition must preserve it while making progress toward termination.", "A proof is incomplete if it establishes preservation but never establishes termination.", "Инвариант задаёт свойство, которое должно сохраняться на выбранных шагах алгоритма.", "Инициализация устанавливает это свойство до начала цикла или рекурсии.", "Каждый переход сохраняет его и приближает выполнение к завершению.", "Доказательство неполно, если оно подтверждает сохранение, но не обосновывает завершение."),
    ("data_provenance", "engineering", "Data provenance records where an artifact came from and how it was produced.", "A source identifier and content hash make later changes detectable.", "Transformation steps should record their inputs and parameters.", "A provenance label is not evidence unless it can be checked against the artifact.", "Происхождение данных фиксирует источник артефакта и способ его получения.", "Идентификатор источника и хэш содержимого позволяют обнаружить последующие изменения.", "Преобразования должны сохранять сведения о входах и параметрах.", "Метка происхождения не является доказательством, пока её нельзя сверить с артефактом."),
    ("control_stability", "control", "A stable feedback loop keeps bounded disturbances from producing unbounded responses.", "Delay and gain affect how quickly the controller reacts to an error.", "A design margin accounts for uncertainty in the plant and its measurements.", "Stability under one operating condition does not establish stability across every condition.", "Устойчивая система с обратной связью не позволяет ограниченным возмущениям вызывать неограниченный отклик.", "Задержка и усиление влияют на скорость реакции регулятора на ошибку.", "Запас устойчивости учитывает неопределённость объекта управления и измерений.", "Устойчивость в одном режиме не доказывает устойчивость во всех режимах."),
    ("robot_force_sensing", "physical_ai", "A force sensor measures interaction between a robot and its environment.", "The measurement depends on sensor calibration and the coordinate frame.", "A controller can use the signal to limit contact force during manipulation.", "A saturated or disconnected sensor cannot justify a claim that contact is safe.", "Датчик силы измеряет взаимодействие робота с окружением.", "Показания зависят от калибровки датчика и системы координат.", "Регулятор использует сигнал, чтобы ограничивать усилие при манипуляции.", "Насыщенный или отключённый датчик не позволяет утверждать, что контакт безопасен."),
    ("resource_scheduling", "computing", "A scheduler assigns limited compute resources to pending work.", "Priority and deadline rules determine which task receives service first.", "Backpressure can prevent producers from overwhelming a slower consumer.", "A queue length alone does not reveal whether the system is meeting its service objective.", "Планировщик распределяет ограниченные вычислительные ресурсы между ожидающими задачами.", "Правила приоритета и срока задают очерёдность обслуживания.", "Обратное давление не даёт производителям перегружать более медленного потребителя.", "Длина очереди сама по себе не показывает, достигает ли система цели обслуживания."),
    ("error_recovery", "engineering", "A recovery procedure starts from a known state after a detected failure.", "The system records which operations completed before the interruption.", "Idempotent steps can be retried without applying the same effect twice.", "Recovery must stop when the saved state cannot be verified.", "Процедура восстановления начинает работу из известного состояния после обнаруженного сбоя.", "Система фиксирует, какие операции завершились до прерывания.", "Идемпотентные шаги можно повторить без двойного применения эффекта.", "Восстановление должно остановиться, если сохранённое состояние нельзя проверить."),
    ("formal_specification", "engineering", "A specification states observable requirements and the conditions under which they apply.", "An implementation can be checked against those requirements with explicit evidence.", "Ambiguous terms should be resolved before they control a safety-relevant action.", "A statement of intent alone does not define a testable acceptance condition.", "Спецификация задаёт наблюдаемые требования и условия их применения.", "Реализацию проверяют по этим требованиям с помощью явных свидетельств.", "Неоднозначные термины следует уточнить до того, как они начнут управлять важным действием.", "Одного описания намерения недостаточно для проверяемого условия приёмки."),
    ("language_register", "language", "A language register adapts vocabulary and sentence structure to a setting and audience.", "A technical explanation can define a term before using it in a narrower sense.", "Consistent terminology helps readers track the same concept across paragraphs.", "Simplifying a sentence should preserve its factual conditions and limits.", "Речевой регистр приспосабливает лексику и построение фраз к ситуации и аудитории.", "Техническое объяснение может определить термин до его употребления в узком смысле.", "Последовательная терминология помогает отслеживать одну мысль в разных абзацах.", "Упрощение фразы должно сохранять фактические условия и ограничения."),
    ("translation_equivalence", "language", "A translation preserves the intended meaning rather than the source sentence's word order.", "Grammar and idiom can require a different construction in the target language.", "A technical term should remain consistent when its meaning is stable across the text.", "A fluent sentence is still inaccurate if it changes a condition or negation.", "Перевод сохраняет исходный смысл, а не порядок слов исходного предложения.", "Грамматика и идиоматика могут требовать другой конструкции на целевом языке.", "Технический термин следует переводить последовательно, если его значение не меняется.", "Беглая фраза всё равно неточна, если в ней изменено условие или отрицание."),
    ("measurement_units", "physics", "A physical quantity is reported with a value and a unit.", "Unit conversion changes the numerical representation while preserving the quantity.", "Dimensional analysis can reveal an equation that combines incompatible quantities.", "A bare number is insufficient when its unit affects the interpretation.", "Физическую величину записывают вместе со значением и единицей измерения.", "Перевод единиц меняет числовую запись, сохраняя саму величину.", "Проверка размерностей выявляет уравнение с несовместимыми величинами.", "Одного числа недостаточно, если единица меняет его смысл."),
    ("robot_trajectory", "physical_ai", "A trajectory specifies a sequence of positions over time.", "Velocity and acceleration constraints limit which trajectories a robot can follow.", "A planner checks the path against obstacles and the robot's current state.", "A trajectory must be replanned when new observations invalidate its safety assumptions.", "Траектория задаёт последовательность положений во времени.", "Ограничения скорости и ускорения сужают множество доступных роботу траекторий.", "Планировщик сверяет путь с препятствиями и текущим состоянием робота.", "Траекторию следует пересчитать, если новые наблюдения нарушили предположения о безопасности."),
    ("uncertainty_propagation", "reasoning", "Uncertainty in an input can affect the range of possible outputs.", "A propagation rule depends on the relationship between the input and output quantities.", "A conservative bound can be useful when a precise distribution is unavailable.", "Ignoring uncertainty does not make an estimate more certain.", "Неопределённость входа влияет на диапазон возможных выходов.", "Правило переноса зависит от связи между входными и выходными величинами.", "Консервативная граница полезна, если точное распределение неизвестно.", "Игнорирование неопределённости не делает оценку более надёжной."),
    ("access_control", "computing", "Access control decides which principal may perform an operation on a resource.", "A policy evaluates identity, requested action, and relevant context.", "The system should deny access when required authorization evidence is missing.", "Authentication establishes an identity claim but does not by itself grant every permission.", "Контроль доступа определяет, кому разрешено выполнять действие над ресурсом.", "Политика проверяет субъект, запрошенную операцию и релевантный контекст.", "Система должна отказать, если необходимых подтверждений полномочий нет.", "Аутентификация подтверждает заявление об идентичности, но сама по себе не выдаёт все права."),
    ("energy_efficiency", "engineering", "Energy efficiency compares useful output with energy supplied under stated conditions.", "A system boundary determines which inputs and losses enter the calculation.", "Measurements should cover the same operating interval for a fair comparison.", "Efficiency figures from different boundaries cannot be compared without adjustment.", "Энергоэффективность сопоставляет полезный выход с подведённой энергией при заданных условиях.", "Граница системы определяет, какие входы и потери входят в расчёт.", "Для корректного сравнения измеряют одинаковые интервалы работы.", "Показатели для разных границ нельзя сравнивать без пересчёта."),
    ("decision_constraints", "reasoning", "A constrained decision selects an available action that satisfies required conditions.", "The objective ranks feasible choices according to a stated preference.", "A hard constraint removes an action even when its score is otherwise high.", "If no feasible action remains, the system should report that fact instead of inventing one.", "Решение с ограничениями выбирает доступное действие, удовлетворяющее обязательным условиям.", "Целевая функция ранжирует допустимые варианты согласно заданному предпочтению.", "Жёсткое ограничение исключает действие даже при высокой оценке.", "Если допустимых действий не осталось, система должна сообщить об этом, а не выдумывать вариант."),
    ("versioned_interfaces", "engineering", "An interface contract defines the inputs and outputs expected by connected components.", "A version change can alter meaning even when field names remain unchanged.", "Compatibility checks should test behavior as well as data shape.", "An undocumented assumption becomes a failure point when components evolve independently.", "Контракт интерфейса определяет входы и выходы связанных компонентов.", "Изменение версии может изменить смысл, даже если названия полей сохранились.", "Проверка совместимости должна охватывать поведение, а не только форму данных.", "Недокументированное предположение становится точкой отказа при независимом развитии компонентов."),
)
)

_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authored_text(concept: tuple[str, ...], language: str, ordinal: int) -> str:
    # The proposition order is preserved. Variants change the authored framing
    # and connective language without shuffling causal claims out of context.
    en = concept[2:6]
    ru = concept[6:10]
    claims = en if language == "en" else ru
    if ordinal >= 128:
        raise ValueError("each authored concept supports at most 128 distinct discourse forms")
    text = " ".join(claims)
    if language == "en":
        lead = ("In {domain}, consider the following principle. ",
                "A useful account of {domain} begins with this distinction. ",
                "For {domain}, the mechanism and its limits belong together. ",
                "This explanation of {domain} connects evidence with a consequence. ",
                "When reasoning about {domain}, keep the stated conditions in view. ",
                "The following account describes {domain} without assuming more than the evidence supports. ",
                "A practical treatment of {domain} separates the rule from its limitation. ",
                "In an engineering discussion of {domain}, context matters. ",
                "A reliable explanation of {domain} identifies both the mechanism and the evidence. ",
                "The key question in {domain} is what the available information can establish. ",
                "A system-level view of {domain} includes its inputs, state, and operating limits. ",
                "When applying {domain}, distinguish a measured fact from an assumption. ",
                "The central ideas in {domain} are linked by conditions and observable effects. ",
                "An accurate account of {domain} states what can fail as well as what can work. ",
                "For a careful analysis of {domain}, follow the relation from cause to outcome. ",
                "A complete description of {domain} names the assumptions behind each conclusion. ")
        endings = ("", " These limits should remain part of the conclusion.",
                   " The stated conditions determine how far the conclusion applies.",
                   " This distinction matters when the system is used in practice.",
                   " The explanation is valid only within the stated operating conditions.",
                   " A sound decision therefore depends on both the evidence and its limits.",
                   " The same rule should be checked again when the surrounding state changes.",
                   " No stronger conclusion follows without additional observations.")
    else:
        lead = ("В теме «{domain}» важно учитывать следующее правило. ",
                "Разговор о теме «{domain}» следует начинать с этого различия. ",
                "Для «{domain}» механизм и его ограничения нужно рассматривать вместе. ",
                "Это объяснение темы «{domain}» связывает условие с последствием. ",
                "При анализе темы «{domain}» необходимо учитывать заданные условия. ",
                "Ниже тема «{domain}» описана без выводов, которых не подтверждают данные. ",
                "Практический разбор темы «{domain}» отделяет правило от его ограничений. ",
                "В инженерном обсуждении темы «{domain}» существенен контекст. ",
                "Надёжное объяснение темы «{domain}» указывает и механизм, и подтверждающие данные. ",
                "Главный вопрос в теме «{domain}» — что именно подтверждает доступная информация. ",
                "Системный взгляд на тему «{domain}» учитывает входы, состояние и рабочие ограничения. ",
                "Применяя правило из темы «{domain}», отделяйте измеренный факт от предположения. ",
                "Основные положения темы «{domain}» связаны условиями и наблюдаемыми последствиями. ",
                "Точное описание темы «{domain}» говорит не только о возможностях, но и об отказах. ",
                "Для аккуратного анализа темы «{domain}» проследите связь от причины к результату. ",
                "Полное описание темы «{domain}» называет предположения, на которых основан каждый вывод. ")
        endings = ("", " Эти ограничения необходимо учитывать в итоговом выводе.",
                   " Заданные условия определяют область применимости вывода.",
                   " Это различие важно при практическом использовании системы.",
                   " Объяснение верно только в указанных рабочих условиях.",
                   " Поэтому обоснованное решение зависит и от свидетельств, и от их ограничений.",
                   " То же правило следует проверить повторно, если состояние окружения изменилось.",
                   " Более сильный вывод потребовал бы дополнительных наблюдений.")
    domain = concept[1]
    if language == "ru":
        domain = {"control": "управление", "physical_ai": "физический ИИ", "reasoning": "рассуждение",
                  "instruction_following": "выполнение инструкций", "computing": "вычисления",
                  "language": "язык", "engineering": "инженерия", "physics": "физика"}[domain]
    lead_index = ordinal % len(lead)
    ending_index = ordinal // len(lead)
    return lead[lead_index].format(domain=domain) + text + endings[ending_index]


def iter_records(root_seed: int, records_per_concept: int) -> Iterator[dict]:
    if type(root_seed) is not int or root_seed < 0 or type(records_per_concept) is not int or records_per_concept < 1:
        raise ValueError("root_seed and records_per_concept must be non-negative/positive integers")
    if records_per_concept > 128:
        raise ValueError("records_per_concept exceeds authored distinct discourse capacity (128)")
    for concept in CONCEPTS:
        family_id = f"language.{concept[0]}.v2"
        for language in LANGUAGES:
            for ordinal in range(records_per_concept):
                seed_payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{language}\0{ordinal}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
                text = _authored_text(concept, language, ordinal)
                body = {"schema": RECORD_SCHEMA, "record_id": "",
                        "text": text, "language": language, "domain": concept[1],
                        "family_id": family_id, "generator_version": ENGINE_VERSION, "seed": seed,
                        "ordinal": ordinal, "purpose": "tokenizer_training_only",
                        "provenance": "NULLXES authored bilingual concept propositions"}
                body["record_id"] = _sha_bytes(json.dumps(
                    [ENGINE_VERSION, root_seed, family_id, language, ordinal], separators=(",", ":")
                ).encode())
                body["text_sha256"] = _sha_bytes(text.encode("utf-8"))
                yield body


def _concept_records(args: tuple[tuple[str, ...], int, int]) -> list[dict]:
    concept, root_seed, records_per_concept = args
    return list(iter_records_for_concept(concept, root_seed, records_per_concept))


def iter_records_for_concept(concept: tuple[str, ...], root_seed: int,
                             records_per_concept: int) -> Iterator[dict]:
    family_id = f"language.{concept[0]}.v2"
    for language in LANGUAGES:
        for ordinal in range(records_per_concept):
            seed_payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{language}\0{ordinal}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
            text = _authored_text(concept, language, ordinal)
            body = {"schema": RECORD_SCHEMA, "record_id": "",
                    "text": text, "language": language, "domain": concept[1],
                    "family_id": family_id, "generator_version": ENGINE_VERSION, "seed": seed,
                    "ordinal": ordinal, "purpose": "tokenizer_training_only",
                    "provenance": "NULLXES authored bilingual concept propositions"}
            body["record_id"] = _sha_bytes(json.dumps(
                [ENGINE_VERSION, root_seed, family_id, language, ordinal], separators=(",", ":")
            ).encode())
            body["text_sha256"] = _sha_bytes(text.encode("utf-8"))
            yield body


def iter_records_parallel(root_seed: int, records_per_concept: int,
                          workers: int) -> Iterator[dict]:
    if type(root_seed) is not int or root_seed < 0 \
            or type(records_per_concept) is not int or not 1 <= records_per_concept <= 128:
        raise ValueError("root_seed must be non-negative and records_per_concept must be in [1, 128]")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        for concept in CONCEPTS:
            yield from iter_records_for_concept(concept, root_seed, records_per_concept)
        return
    jobs = ((concept, root_seed, records_per_concept) for concept in CONCEPTS)
    with ProcessPoolExecutor(max_workers=min(workers, len(CONCEPTS)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for rows in pool.map(_concept_records, jobs, chunksize=1):
            yield from rows


def write_corpus(path: Path, root_seed: int, records_per_concept: int,
                 max_sentence_bytes: int, workers: int = 1) -> dict:
    audit_path = path.with_suffix(".audit.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    domains = {language: Counter() for language in LANGUAGES}
    words = {language: set() for language in LANGUAGES}
    chars = {language: 0 for language in LANGUAGES}
    byte_lengths = {language: 0 for language in LANGUAGES}
    if path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer-only corpus artifact: {path}")
    created_artifact = False
    try:
        with closing(sqlite3.connect(audit_path)) as db:
            db.execute("CREATE TABLE texts (sha TEXT PRIMARY KEY)")
            output_stream = path.open("xb")
            created_artifact = True
            with output_stream as out:
                for row in iter_records_parallel(root_seed, records_per_concept, workers):
                    encoded_text = row["text"].encode("utf-8")
                    if len(encoded_text) > max_sentence_bytes:
                        raise ValueError(f"{row['record_id']}: text exceeds tokenizer max_sentence_bytes")
                    try:
                        db.execute("INSERT INTO texts VALUES (?)", (row["text_sha256"],))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate tokenizer-only text: {row['record_id']}") from exc
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                    language = row["language"]
                    counts[language] += 1
                    domains[language][row["domain"]] += 1
                    words[language].update(token.casefold() for token in _WORD.findall(row["text"]))
                    chars[language] += len(row["text"])
                    byte_lengths[language] += len(encoded_text)
                out.flush()
                os.fsync(out.fileno())
            db.commit()
    except Exception:
        if created_artifact:
            path.unlink(missing_ok=True)
        raise
    finally:
        audit_path.unlink(missing_ok=True)
    stats = {"records_by_language": dict(sorted(counts.items())),
             "records_by_domain": {key: dict(sorted(value.items())) for key, value in domains.items()},
             "unique_casefolded_word_forms_by_language": {key: len(value) for key, value in words.items()},
             "characters_by_language": chars, "utf8_bytes_by_language": byte_lengths,
             "exact_duplicate_text_records": 0, "record_count": sum(counts.values()),
             "records_per_concept": records_per_concept}
    return stats
