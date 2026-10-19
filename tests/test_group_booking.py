import asyncio
import unittest
from unittest import mock

from agent import group_booking as G
from agent.tools import BookingTools

COMPANY = {
    "company": {},
    "services": [{"id": "18", "naziv": "Pedikura", "trajanjeMin": 45},
                 {"id": "27", "naziv": "Refleksna masaža stopal", "trajanjeMin": 60}],
    "employeesByServiceId": {"18": ["1", "2"], "27": ["1", "2"]},
    "employees_ui": [{"id": "1", "label": "Maja Hribar"}, {"id": "2", "label": "Luka Dobrovoljec"}],
    "resursi": [{"id": 4, "naziv": "Miza", "kolicina": 1, "kapaciteta": 1}],
    "storitveResursiMap": {},
}
DATE = "2030-01-10"


def run(coro):
    return asyncio.run(coro)


class Ctx:
    pass


class GroupSearch(unittest.TestCase):
    def setUp(self):
        self.bt = BookingTools("salon", COMPANY)
        self.g = G.GroupBooking(self.bt)

    def _slots(self, by_employee):
        async def fake(**kw):
            emp = kw.get("employee_id")
            times = by_employee.get(emp) if emp else sorted({t for v in by_employee.values() for t in v})
            return {"success": True, "slots": {DATE: times}}
        return fake

    def test_one_after_another(self):
        fake = self._slots({"1": ["14:00", "15:00", "16:00"], "2": ["14:00", "15:00", "16:00"]})
        with mock.patch.object(G.booking_client, "get_slots", fake):
            r = run(self.g.find_group_times(Ctx(), ["18", "18"], ["", ""], DATE, False))
        first = r["options"][0]["people"]
        self.assertEqual([p["time"] for p in first], ["14:00", "15:00"])  # 14:45 rounds up to the next free time
        self.assertEqual(first[1]["say"], "ob treh popoldan")

    def test_together_needs_two_employees(self):
        fake = self._slots({"1": ["10:00", "11:00"], "2": ["11:00"]})
        with mock.patch.object(G.booking_client, "get_slots", fake):
            r = run(self.g.find_group_times(Ctx(), ["27", "27"], ["", ""], DATE, True))
        self.assertEqual([o["people"][0]["time"] for o in r["options"]], ["11:00"])
        emps = {p["employee"] for p in r["options"][0]["people"]}
        self.assertEqual(emps, {"Maja", "Luka"})

    def test_together_blocked_by_single_resource(self):
        company = {**COMPANY, "storitveResursiMap": {"27": [4]}}
        g = G.GroupBooking(BookingTools("salon", company))
        fake = self._slots({"1": ["11:00"], "2": ["11:00"]})
        with mock.patch.object(G.booking_client, "get_slots", fake):
            r = run(g.find_group_times(Ctx(), ["27", "27"], ["", ""], DATE, True))
        self.assertEqual(r["options"], [])


class GroupBook(unittest.TestCase):
    def test_rollback_when_second_fails(self):
        cancelled = []

        async def cancel(row_id):
            cancelled.append(row_id)
            return True

        bt = BookingTools("salon", COMPANY)
        g = G.GroupBooking(bt, cancel_created=cancel)
        g._options = [{"slots": [{"time": "14:00", "employee_id": "1"}, {"time": "15:00", "employee_id": "2"}]}]
        g._date = DATE
        g._people = [{"service_id": "18", "employee_id": "1", "eligible": ["1", "2"], "duration": 45},
                     {"service_id": "18", "employee_id": "2", "eligible": ["1", "2"], "duration": 45}]
        calls = []

        async def check(**kw):
            return {"available": True}

        async def create(**kw):
            calls.append(kw)
            if len(calls) == 2:
                return {"success": False, "error": "slot_taken"}
            return {"success": True, "terminId": "OB-1", "terminRowId": 77}

        with mock.patch.object(G.booking_client, "check_slots", check), \
                mock.patch.object(G.booking_client, "create_booking", create):
            r = run(g.book_group(Ctx(), 0, "Ana", "Novak", "041 222 333", ["Eva"]))
        self.assertFalse(r["success"])
        self.assertEqual(cancelled, [77])
        self.assertIsNone(bt.created_termin_id)
        self.assertEqual(calls[1]["first_name"], "Eva")
        self.assertEqual(calls[0]["phone"], "+38641222333")


class SpokenTime(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(G.spoken_sl_time("12:00"), "ob dvanajstih")
        self.assertEqual(G.spoken_sl_time("13:00"), "ob enih popoldan")
        self.assertEqual(G.spoken_sl_time("09:30"), "ob pol desetih dopoldan")
        self.assertIsNone(G.spoken_sl_time("10:15"))


if __name__ == "__main__":
    unittest.main()
