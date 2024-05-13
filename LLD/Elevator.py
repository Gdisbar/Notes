"""
Initialization:
-----------------------------------------------
Each elevator is initialized with an ID and a current floor (initially set to 0).
The elevator's direction is initially set to IDLE.

Each elevator maintains a priority queue of requests, where requests are 
prioritized based on the current floor.

Request Processing:
-------------------------------------
When a request is added to an elevator, it is inserted into the elevator's 
priority queue.

The elevator then processes its requests in a loop:

        It retrieves the next request from the priority queue.

        If the request's desired floor is greater than the current floor, 
        the elevator sets its direction to UP; if less, it sets the direction 
        to DOWN; if equal, it sets the direction to IDLE.

        The elevator moves to the desired floor, updating its current floor.
        This process continues until all requests in the elevator's queue are 
        processed.

Elevator Controller:
---------------------------------------------------------------
The ElevatorController class manages multiple elevators.

When a request is made, the controller assigns the request to an eligible 
elevator based on its direction and proximity to the request's floor.

If no eligible elevators are available (i.e., all elevators are moving in 
the opposite direction), the controller assigns the request to the nearest 
elevator regardless of its direction.

Simulation:
----------------------------------------
Requests are generated, and each request is added to the elevator controller.

The elevator controller then simulates elevator movements by processing 
requests for each elevator.

"""

from queue import PriorityQueue
from  typing import List

class Request:
    def __init__(self, currentFloor, desiredFloor, direction, location):
        self.currentFloor = currentFloor
        self.desiredFloor = desiredFloor
        self.direction = direction
        self.location = location

    def __lt__(self, other):
        return self.currentFloor < other.currentFloor

class Location:
    INSIDE_ELEVATOR = "INSIDE_ELEVATOR"
    OUTSIDE_ELEVATOR = "OUTSIDE_ELEVATOR"

class Elevator:
    def __init__(self, elevator_id, currentFloor):
        self.elevator_id = elevator_id
        self.currentFloor = currentFloor
        self.direction = Direction.IDLE
        self.requests = PriorityQueue()

    def addRequest(self, request):
        self.requests.put(request)

    def processRequests(self):
        while not self.requests.empty():
            request = self.requests.get()
            if request.desiredFloor > self.currentFloor:
                self.direction = Direction.UP
            elif request.desiredFloor < self.currentFloor:
                self.direction = Direction.DOWN
            else:
                self.direction = Direction.IDLE
                
            print(f"Elevator {self.elevator_id} is processing request from floor {request.currentFloor} to floor {request.desiredFloor}. Moving {self.direction}")
            self.currentFloor = request.desiredFloor

class Direction:
    UP = "UP"
    DOWN = "DOWN"
    IDLE = "IDLE"

class ElevatorController:
    def __init__(self, numElevators):
        self.elevators = [Elevator(i, 0) for i in range(numElevators)]

    def requestElevator(self, request):
        # Filter elevators by direction and select the nearest one
        if request.direction == Direction.UP:
            eligible_elevators = [elevator for elevator in self.elevators if elevator.direction in (Direction.UP, Direction.IDLE)]
        elif request.direction == Direction.DOWN:
            eligible_elevators = [elevator for elevator in self.elevators if elevator.direction in (Direction.DOWN, Direction.IDLE)]
        else:
            eligible_elevators = []

        if eligible_elevators:
            nearestElevator = min(eligible_elevators, key=lambda e: abs(e.currentFloor - request.currentFloor))
            nearestElevator.addRequest(request)
            nearestElevator.processRequests()
        else:
            print("No eligible elevators available to handle the request.")
            nearestElevator = min(self.elevators, key=lambda e: abs(e.currentFloor - request.currentFloor))
            nearestElevator.addRequest(request)
            nearestElevator.processRequests()

    def simulateMovement(self):
        for elevator in self.elevators:
            elevator.processRequests()


if __name__ == "__main__":
    elevatorController = ElevatorController(numElevators=3)

    # Create some requests
    requests = [
        Request(2, 7, Direction.UP, Location.OUTSIDE_ELEVATOR),
        Request(4, 1, Direction.DOWN, Location.OUTSIDE_ELEVATOR),
        Request(0, 8, Direction.UP, Location.OUTSIDE_ELEVATOR),
        Request(5, 3, Direction.DOWN, Location.OUTSIDE_ELEVATOR),
        Request(6, 4, Direction.DOWN, Location.OUTSIDE_ELEVATOR),
        Request(2, 7, Direction.UP, Location.OUTSIDE_ELEVATOR),
        Request(4, 1, Direction.DOWN, Location.OUTSIDE_ELEVATOR),
        Request(0, 8, Direction.UP, Location.OUTSIDE_ELEVATOR),
        Request(5, 3, Direction.DOWN, Location.INSIDE_ELEVATOR),
        Request(6, 4, Direction.DOWN, Location.INSIDE_ELEVATOR),
        Request(2, 7, Direction.UP, Location.INSIDE_ELEVATOR),
        Request(4, 1, Direction.DOWN, Location.INSIDE_ELEVATOR),
    ]

    # Add requests to the elevator controller
    for request in requests:
        elevatorController.requestElevator(request)

    # Simulate elevator movements
    elevatorController.simulateMovement()
