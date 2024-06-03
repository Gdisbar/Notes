from enum import Enum
from datetime import datetime
import time
from dataclasses import dataclass
from typing import Dict,List

class ParkingSlotType(Enum):
    TwoWheeler = 1
    Compact = 2
    Medium = 3
    Large = 4

    def getPriceForParking(self, duration):
        if self == ParkingSlotType.TwoWheeler:
            return duration * 0.05
        elif self == ParkingSlotType.Compact:
            return duration * 0.075
        elif self == ParkingSlotType.Medium:
            return duration * 0.09
        elif self == ParkingSlotType.Large:
            return duration * 0.10

from dataclasses import dataclass, field
from typing import Optional

class VehicleCategory(Enum):
    TwoWheeler = 1
    Hatchback = 2
    Sedan = 3
    SUV = 4
    Bus = 5

@dataclass
class Vehicle:
    vehicle_number: str
    vehicle_category: 'VehicleCategory'  # forward reference to resolve circular import issues

@dataclass
class Ticket:
    ticket_number: str
    start_time: float
    end_time: float
    vehicle: 'Vehicle'
    parking_slot: Optional[ParkingSlotType] = None  # Changed ParkingSlot to ParkingSlotType

    @classmethod
    def create_ticket(cls, vehicle, parking_slot):
        ticket_number = vehicle.vehicle_number + str(time.time())
        return cls(ticket_number=ticket_number,
                   start_time=time.time(),
                   end_time=None,
                   vehicle=vehicle,
                   parking_slot=parking_slot)

@dataclass
class ParkingSlot:
    name: str
    is_available: bool = True
    vehicle: Vehicle = None
    parking_slot_type: ParkingSlotType = None

    def __init__(self, name: str, parking_slot_type: ParkingSlotType):
        self.name = name
        self.parking_slot_type = parking_slot_type

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.is_available = False

    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicle = None
        self.is_available = True


# vehicle = Vehicle(vehicle_number="ABC123", vehicle_category=VehicleCategory.SUV)
# parking_slot = ParkingSlotType.Large
# ticket = Ticket.create_ticket(vehicle, parking_slot)
# print("Ticket Number:", ticket.ticket_number)
# print("Start Time:", datetime.fromtimestamp(ticket.start_time))
# print("Vehicle Number:", ticket.vehicle.vehicle_number)
# print("Parking Slot Type:", ticket.parking_slot.name)
# print("Price for Parking:", ticket.parking_slot.getPriceForParking(2))  # assuming 2 hours duration
# parking_slot_name = "G1A1"
# parking_slot_entry = ParkingSlot(parking_slot_name,parking_slot)
# parking_slot_entry.add_vehicle(vehicle)
# print(f"Parking Slot No : {parking_slot_entry.name}, Number : {vehicle.vehicle_number} & Type : {vehicle.vehicle_category}")


"""
ParkingFloor: Class representing a parking floor.

__init__: Constructor method to initialize the name and parking slots.

get_relevant_slot_for_vehicle_and_park: Method to find and return a relevant 
parking slot for a given vehicle.

pick_correct_slot: Method to determine the correct parking slot type based 
on the vehicle category.

The Vehicle and ParkingSlotType classes are imported from the appropriate modules.

Dict is used for type hinting to specify the type of parking_slots dictionary.

Looping over dictionary items is done using values() to access the values directly.

The VehicleCategory enum is imported separately, assuming it's defined in a separate module.

"""

class ParkingFloor:
    def __init__(self, name: str, parking_slots: Dict[ParkingSlotType, Dict[str, 'ParkingSlot']]):
        self.name = name
        self.parking_slots = parking_slots

    def get_relevant_slot_for_vehicle_and_park(self, vehicle: Vehicle) -> 'ParkingSlot':
        vehicle_category = vehicle.vehicle_category
        parking_slot_type = self.pick_correct_slot(vehicle_category)
        relevant_parking_slots = self.parking_slots.get(parking_slot_type)
        slot = None
        if relevant_parking_slots:
            for parking_slot in relevant_parking_slots.values():
                if parking_slot.is_available:
                    slot = parking_slot
                    slot.add_vehicle(vehicle)
                    break
        return slot

    def pick_correct_slot(self, vehicle_category: VehicleCategory) -> ParkingSlotType:
        if vehicle_category == VehicleCategory.TwoWheeler:
            return ParkingSlotType.TwoWheeler
        elif vehicle_category in [VehicleCategory.Hatchback, VehicleCategory.Sedan]:
            return ParkingSlotType.Compact
        elif vehicle_category == VehicleCategory.SUV:
            return ParkingSlotType.Medium
        elif vehicle_category == VehicleCategory.Bus:
            return ParkingSlotType.Large
        return None


class ParkingLot:
    def __init__(self, name_of_parking_lot: str, parking_floors: List[ParkingFloor]):
        self.name_of_parking_lot = name_of_parking_lot
        self.parking_floors = parking_floors
        self.parking_lot = None

    @classmethod
    def get_instance(cls, name_of_parking_lot: str, parking_floors: List[ParkingFloor]):
        if not cls.parking_lot:
            cls.parking_lot = cls(name_of_parking_lot, parking_floors)
        return cls.parking_lot


    def add_floors(self, name: str, parking_slots: Dict[ParkingSlotType, Dict[str, ParkingSlot]]):
        parking_floor = ParkingFloor(name, parking_slots)
        self.parking_floors.append(parking_floor)

    def remove_floors(self, parking_floor: ParkingFloor):
        self.parking_floors.remove(parking_floor)

    def assign_ticket(self, vehicle: Vehicle) -> Ticket:
        parking_slot = self.get_parking_slot_for_vehicle_and_park(vehicle)
        if not parking_slot:
            return None
        parking_ticket = self.create_ticket_for_slot(parking_slot, vehicle)
        # persist ticket to database
        return parking_ticket

    def scan_and_pay(self, ticket: Ticket) -> float:
        end_time = time.time() * 1000  # in milliseconds
        ticket.parking_slot.remove_vehicle(ticket.vehicle)
        duration = int((end_time - ticket.start_time) / 1000)
        price = ticket.parking_slot.parking_slot_type.get_price_for_parking(duration)
        # persist record to database
        return price

    def create_ticket_for_slot(self, parking_slot: ParkingSlot, vehicle: Vehicle) -> Ticket:
        return Ticket.create_ticket(vehicle, parking_slot)

    def get_parking_slot_for_vehicle_and_park(self, vehicle: Vehicle) -> ParkingSlot:
        parking_slot = None
        for floor in self.parking_floors:
            parking_slot = floor.get_relevant_slot_for_vehicle_and_park(vehicle)
            if parking_slot:
                break
        return parking_slot


if __name__ == "__main__":
    name_of_parking_lot = "Pintosss Parking Lot"
    # address = Address(city="Bangalore", country="India", state="KA")
    
    compact_slots = {
        "C1": ParkingSlot("C1", ParkingSlotType.Compact),
        "C2": ParkingSlot("C2", ParkingSlotType.Compact),
        "C3": ParkingSlot("C3", ParkingSlotType.Compact)
    }
    
    large_slots = {
        "L1": ParkingSlot("L1", ParkingSlotType.Large),
        "L2": ParkingSlot("L2", ParkingSlotType.Large),
        "L3": ParkingSlot("L3", ParkingSlotType.Large)
    }
    
    all_slots = {
        ParkingSlotType.Compact: compact_slots,
        ParkingSlotType.Large: large_slots
    }
    
    parking_floor = ParkingFloor("1", all_slots)
    parking_floors = [parking_floor]
    
    parking_lot = ParkingLot.get_instance(name_of_parking_lot, parking_floors)
    
    vehicle = Vehicle()
    vehicle.vehicle_category = VehicleCategory.Hatchback
    vehicle.vehicle_number = "KA-01-MA-9999"
    
    ticket = parking_lot.assign_ticket(vehicle)
    print("Ticket number:", ticket.ticket_number)
    # Persist the ticket to the database here
    time.sleep(10)
    price = parking_lot.scan_and_pay(ticket)
    print("Price is:", price)
