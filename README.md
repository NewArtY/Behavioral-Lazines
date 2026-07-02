# Код к статье «Лень по поведению» (для публикации на GitHub)

Самодостаточное ядро метода (только `numpy`) + воспроизведение синтетического
примера из статьи. Полные доменные прогоны (рынок труда, релятивистский
электронный ансамбль) используют отдельные пакеты `python_forecast` /
`python_solver` и закрытые данные — в публичный репозиторий пойдёт это ядро и
синтетика, а доменные результаты — ссылкой / отдельным релизом.

## Состав

- `laziness.py` — причинная регуляризация траектории трансфер-оператора:
  `null_space_project`, `smooth_series_nullspace` (жёсткая, сохраняет `A·P`
  точно), `smooth_series_ewma` (мягкая), `trajectory_energy`.
- `baselines.py` — OT/энтропийные baseline'ы: `sinkhorn` (энтропийный
  оптимальный транспорт), `coupling_retention`, `transport_cost`.
- `synthetic_demo.py` — §5.3: контролируемая истина, вращающиеся `A(t)`; null
  помогает, ewma при реальном дрейфе вредит.
- `sinkhorn_demo.py` — §2.3/§3.4: энтропийный OT-коупл; churn = настраиваемый
  выбор (`reg`) → OT параметризует приор, а не устраняет.
- `phase_map.py` — §5.3/Рис. 4: карта применимости (delta × rho); null-space
  безопасна (ratio≤1, не зависит от тренда), ewma — зона вреда. Пишет CSV в
  `experiments/`.
- `lambda_cv.py` — §4.3: причинный выбор λ (inner-CV) бьёт raw и ~совпадает с
  оракулом для null-space.
- `experiments/` — доменные прогоны (нужны закрытые данные/солвер) + CSV-сетки
  карты + `timing.log`.
- `requirements.txt` — зависимости (`numpy`).

## Запуск (самодостаточные, только numpy)

```bash
pip install -r requirements.txt
python synthetic_demo.py      # §5.3 механизм
python sinkhorn_demo.py       # §2.3 OT параметризует приор
python phase_map.py           # §5.3 карта применимости (CSV в experiments/)
python lambda_cv.py           # §4.3 причинный выбор lambda
```

Все детерминированы по сидам. Время прогонов — в `experiments/timing.log`
(всё CPU, < 10 с каждый; GPU не нужен).

## Планируемая структура публичного репозитория

```
behavioral-laziness/
  laziness.py
  synthetic_demo.py
  figures/            генераторы рисунков (matplotlib)
  experiments/        обёртки доменных прогонов (ссылки на данные)
  tests/              перенести test_laziness.py
  LICENSE             MIT
  CITATION.cff        метаданные для цитирования
  .zenodo.json        метаданные для Zenodo DOI
```

## Лицензия

MIT — см. [`LICENSE`](LICENSE). © 2026 Невечеря А. П.

## Цитирование и DOI (Zenodo)

Код архивируется на Zenodo с присвоением DOI. Метаданные архива берутся из
[`.zenodo.json`](.zenodo.json), метаданные для менеджеров цитирования — из
[`CITATION.cff`](CITATION.cff) (GitHub показывает кнопку «Cite this repository»).

Автор и аффилиация уже заполнены в метаданных (Невечеря А. П., ORCID
[0000-0001-6736-4691](https://orcid.org/0000-0001-6736-4691), Кубанский
государственный университет). Остаются только поля, которые физически нельзя
проставить до соответствующего события (URL репозитория, DOI Zenodo, выходные
данные статьи) — они помечены `TODO` в `CITATION.cff`.

Порядок получения DOI:
1. Залить репозиторий на GitHub, включить его в Zenodo (zenodo.org → GitHub → *toggle on*).
   Вписать URL репозитория в `repository-code`/`url` (`CITATION.cff`).
2. Создать GitHub Release (тег, напр. `v1.0.0`) — Zenodo автоматически заархивирует и выдаст DOI.
3. Вписать полученный **concept DOI** обратно в `CITATION.cff` (поле `doi:`) и добавить бейдж DOI в этот README.
4. После публикации/принятия статьи — раскомментировать блок `preferred-citation`
   в `CITATION.cff` (том/номер/страницы/DOI уже подготовлены с реальным названием
   и журналом), чтобы цитирование вело на статью, а не на архив кода.

## TODO
- [ ] Перенести `test_laziness.py` сюда.
- [ ] Генераторы рисунков 1–4.
- [x] LICENSE (MIT) + CITATION.cff + .zenodo.json.
- [x] Автор, ORCID, аффилиация, e-mail, ключевые слова, двуязычное название — заполнены.
- [ ] URL репозитория — после заливки на GitHub.
- [ ] DOI Zenodo — после первого релиза.
- [ ] Раскомментировать `preferred-citation` (статья) — после публикации (выходные данные подготовлены).
