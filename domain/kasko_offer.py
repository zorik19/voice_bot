from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# Модель данных для проактивного обзвона "клиент посчитал КАСКО и не оплатил".
# Повторяет структуру ответа внутреннего API (см. постановку, раздел 2):
#   2.1 — расчёт клиента из ССО 2.0 (что клиент сам посчитал на сайте)
#   2.2 — оптимальные условия по модели (businessData.modelResponse)
#   2.3 — otherParamsList: таблица уже готовых ценовых точек
#
# Точная топ-level схема ответа API пока не согласована с бэкендом — маппинг
# в load_offer_from_api ниже best-effort по названиям полей из постановки,
# уточнить при реальной интеграции.

_MONTHS_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

_DEDUCTIBLE_TYPE_RU = {
    "uslovnaya": "условная франшиза",
    "bezuslovnaya": "безусловная франшиза",
    "uslovno-bezuslovnaya": "смешанная франшиза",
}


def format_date_ru(d: date) -> str:
    return f"{d.day} {_MONTHS_RU[d.month - 1]} {d.year} года"


def deductible_type_ru(code: str | None) -> str:
    if not code:
        return "без франшизы"
    return _DEDUCTIBLE_TYPE_RU.get(code.lower(), code)


@dataclass(slots=True)
class PrimaryCalculation:
    """2.1 — что клиент сам посчитал на сайте и не оплатил."""
    fio: str
    vehicle_mark: str
    vehicle_model: str
    program: str
    current_price: float
    current_insurance_sum: float
    current_deductible_type: str | None   # None, если франшизы нет
    current_deductible_sum: float          # -1, если франшизы нет
    active_date: date
    end_date: date
    calculation_expiry: date
    risks: list[str] = field(default_factory=list)

    @property
    def has_deductible(self) -> bool:
        return bool(self.current_deductible_type) and self.current_deductible_sum > 0

    @property
    def vehicle(self) -> str:
        return f"{self.vehicle_mark} {self.vehicle_model}"


@dataclass(slots=True)
class OptimumCondition:
    """2.2 — оптимальное сочетание сумма/франшиза/цена по модели.
    loss_frequency, loss_amount, ccu — внутренняя аналитика, клиенту НЕ озвучиваются."""
    insurance_sum: float
    deductible_type: str
    deductible_sum: float
    price: float
    loss_frequency: float
    loss_amount: float
    ccu: float


@dataclass(slots=True)
class AlternativeOffer:
    """2.3 — одна строка otherParamsList. conversion/net_income/ccu — внутренние
    метрики для аналитики, клиенту не озвучиваются; бот оперирует только price."""
    price: float
    conversion: float
    net_income: float
    ccu: float


@dataclass(slots=True)
class ClientOffer:
    primary: PrimaryCalculation
    optimum: OptimumCondition | None
    alternatives: list[AlternativeOffer]  # по возрастанию цены (как отдаёт система)


def load_offer_from_api(payload: dict[str, Any]) -> ClientOffer:
    """Разбор ответа внутреннего API в ClientOffer. Формат полей — по постановке
    (раздел 2); точный top-level конверт уточняется при реальной интеграции."""
    primary = PrimaryCalculation(
        fio=payload["fio"],
        vehicle_mark=payload["vehicleMark"],
        vehicle_model=payload["vehicleModel"],
        program=payload["program"],
        current_price=float(payload["currentPrice"]),
        current_insurance_sum=float(payload["currentInsuranceSum"]),
        current_deductible_type=payload.get("currentDeductibleType"),
        current_deductible_sum=float(payload.get("currentDeductibleSum", -1)),
        active_date=date.fromisoformat(payload["activeDate"]),
        end_date=date.fromisoformat(payload["endDate"]),
        calculation_expiry=date.fromisoformat(payload["calculationExpiry"]),
        risks=list(payload.get("risk", [])),
    )

    model_response = payload.get("businessData", {}).get("modelResponse")
    optimum = None
    if model_response:
        optimum = OptimumCondition(
            insurance_sum=float(model_response["optimumInsCondition"]),
            deductible_type=model_response["optimumDeducTypeCond"],
            deductible_sum=float(model_response["optimumDeducValCondition"]),
            price=float(model_response["optimumPriceCondition"]),
            loss_frequency=float(model_response.get("lossFreqForOptimumCondition", 0)),
            loss_amount=float(model_response.get("lossAmountForOptimumCondition", 0)),
            ccu=float(model_response.get("ccuForOptimumCondition", 0)),
        )

    alternatives = [
        AlternativeOffer(
            price=float(row["price"]),
            conversion=float(row.get("conversion", 0)),
            net_income=float(row.get("netIncome", 0)),
            ccu=float(row.get("ccu", 0)),
        )
        for row in payload.get("otherParamsList", [])
    ]

    return ClientOffer(primary=primary, optimum=optimum, alternatives=alternatives)


def sample_offer() -> ClientOffer:
    """Фикстура для демо/прототипа — без реального API."""
    today = date(2026, 8, 17)
    primary = PrimaryCalculation(
        fio="Иванов Пётр Сергеевич",
        vehicle_mark="Kia",
        vehicle_model="Sportage",
        program="Классика",
        current_price=89_500,
        current_insurance_sum=2_100_000,

        current_deductible_type=None,
        current_deductible_sum=-1,
        active_date=today + timedelta(days=5),
        end_date=today + timedelta(days=370),
        calculation_expiry=today + timedelta(days=14),
        risks=["Ущерб", "Угон", "Полная гибель", "Стихийные бедствия", "Действия третьих лиц"],
    )
    optimum = OptimumCondition(
        insurance_sum=2_100_000,
        deductible_type="Uslovno-Bezuslovnaya",
        deductible_sum=30_000,
        price=71_200,
        loss_frequency=0.18,
        loss_amount=145_000,
        ccu=0.62,
    )
    alternatives = [
        AlternativeOffer(price=68_900, conversion=0.21, net_income=6_100, ccu=0.58),
        AlternativeOffer(price=71_200, conversion=0.27, net_income=7_400, ccu=0.62),
        AlternativeOffer(price=76_400, conversion=0.34, net_income=9_800, ccu=0.68),
        AlternativeOffer(price=82_300, conversion=0.41, net_income=11_900, ccu=0.74),
        AlternativeOffer(price=89_500, conversion=0.46, net_income=13_200, ccu=0.79),
        AlternativeOffer(price=95_100, conversion=0.51, net_income=14_600, ccu=0.83),
    ]
    return ClientOffer(primary=primary, optimum=optimum, alternatives=alternatives)


def format_offer_for_prompt(offer: ClientOffer) -> str:
    """Текстовый блок с данными клиента для системного промпта. Внутренняя
    аналитика (conversion/netIncome/ccu/lossFreq/lossAmount) сюда НЕ попадает —
    бот не должен иметь возможность случайно её озвучить."""
    p = offer.primary
    lines = [
        f"ФИО клиента: {p.fio}",
        f"Автомобиль: {p.vehicle}",
        f"Программа: {p.program}",
        f"Риски в расчёте: {', '.join(p.risks) if p.risks else 'не указаны'}",
        f"Текущий расчёт клиента (то, что он сам считал на сайте и не оплатил):",
        f"  цена {p.current_price:.0f} руб., страховая сумма {p.current_insurance_sum:.0f} руб., "
        f"{deductible_type_ru(p.current_deductible_type)}"
        + (f" {p.current_deductible_sum:.0f} руб." if p.has_deductible else ""),
        f"  период действия: с {format_date_ru(p.active_date)} по {format_date_ru(p.end_date)}",
        f"  расчёт актуален до {format_date_ru(p.calculation_expiry)}",
    ]

    if offer.optimum:
        o = offer.optimum
        better = o.price < p.current_price
        lines.append(
            "Оптимальный вариант по модели"
            + (" (дешевле текущего расчёта — стоит предложить отдельно)" if better else "")
            + f": страховая сумма {o.insurance_sum:.0f} руб., {deductible_type_ru(o.deductible_type)} "
            f"{o.deductible_sum:.0f} руб., цена {o.price:.0f} руб."
        )

    if offer.alternatives:
        def _row(a: AlternativeOffer) -> str:
            delta = p.current_price - a.price
            if delta > 0:
                tag = f"дешевле текущего расчёта на {delta:.0f} руб."
            elif delta < 0:
                tag = f"дороже текущего расчёта на {-delta:.0f} руб. (например, доплата за ремонт у дилера)"
            else:
                tag = "равна текущему расчёту"
            return f"{a.price:.0f} руб. ({tag})"

        table = "; ".join(_row(a) for a in offer.alternatives)
        lines.append(
            "Готовая таблица альтернативных цен для этого клиента (строго по возрастанию "
            "цены; дельта в скобках посчитана относительно ИСХОДНОГО расчёта клиента выше, "
            "не относительно того, что ты уже успела назвать в разговоре — это надо "
            "отслеживать самой по ходу диалога. Конкретные страховая сумма и франшиза для "
            "каждой точки системой не переданы — используй как ценовые варианты под запрос "
            "клиента, например 'дешевле на N рублей'): " + table
        )

    return "\n".join(lines)