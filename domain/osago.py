from __future__ import annotations

from typing import Any

# Упрощённая модель для демо: премия = ТБ x КТ x КБМ x КВС x КМ.
# Коэффициенты примерные, структура повторяет реальную методику ЦБ.

BASE_RATE = 5_500  # ТБ, руб — середина тарифного коридора для физлиц

REGION_KT = {
    "москва": 1.8,
    "санкт-петербург": 1.64,
    "московская область": 1.56,
    "казань": 1.8,
    "екатеринбург": 1.72,
    "новосибирск": 1.63,
    "прочие": 1.0,
}


def _kt(region: str) -> float:
    return REGION_KT.get(region.lower().strip(), REGION_KT["прочие"])


def _km(power_hp: int) -> float:
    if power_hp <= 50:
        return 0.6
    if power_hp <= 70:
        return 1.0
    if power_hp <= 100:
        return 1.1
    if power_hp <= 120:
        return 1.2
    if power_hp <= 150:
        return 1.4
    return 1.6


def _kvs(age: int, experience_years: int) -> float:
    """Возраст-стаж: молодой без стажа — дорого, взрослый со стажем — скидка."""
    if age < 22:
        return 1.92 if experience_years < 3 else 1.65
    if age < 25:
        return 1.77 if experience_years < 3 else 1.13
    if age < 35:
        return 1.17 if experience_years < 3 else 0.95
    return 1.08 if experience_years < 3 else 0.9


def calculate_osago(
    region: str,
    power_hp: int,
    age: int,
    experience_years: int,
    kbm: float = 1.0,
) -> dict[str, Any]:
    kt, km, kvs = _kt(region), _km(power_hp), _kvs(age, experience_years)
    premium = round(BASE_RATE * kt * kbm * kvs * km)
    return {
        "premium_rub": premium,
        "details": {"base": BASE_RATE, "kt": kt, "kbm": kbm, "kvs": kvs, "km": km},
    }


# Схема тула (anthropic-совместимый формат, наш внутренний стандарт)
OSAGO_TOOL_SCHEMA = {
    "name": "calculate_osago",
    "description": (
        "Рассчитать примерную стоимость полиса ОСАГО. Вызывай, когда собраны "
        "все данные: регион, мощность двигателя в л.с., возраст водителя, "
        "стаж вождения в годах. КБМ передавай, если клиент его знает, иначе 1.0."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "Город/регион регистрации"},
            "power_hp": {"type": "integer", "description": "Мощность двигателя, л.с."},
            "age": {"type": "integer", "description": "Возраст водителя, лет"},
            "experience_years": {"type": "integer", "description": "Стаж вождения, лет"},
            "kbm": {
                "type": "number",
                "description": "Коэффициент бонус-малус, по умолчанию 1.0",
                "default": 1.0,
            },
        },
        "required": ["region", "power_hp", "age", "experience_years"],
    },
}
