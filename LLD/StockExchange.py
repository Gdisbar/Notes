import heapq
from abc import ABC, abstractmethod

class OrderInterface(ABC):
    @abstractmethod
    def __init__(self, price, orderNo, quantity):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __ge__(self, other):
        pass

    # @abstractmethod
    # def __eq__(self, other):
    #     pass

class Orders(OrderInterface):
    def __init__(self, price, orderNo, quantity):
        self.price = price
        self.orderNo = orderNo
        self.quantity = quantity
    
    def __lt__(self, other):
        if self.price != other.price:
            return self.price < other.price
        return self.orderNo < other.orderNo

    def __ge__(self, other):
        return self > other or self == other

    # def __eq__(self, other):
    #     return self.price == other.price and self.orderNo == other.orderNo



class OrderBook(object):
    def __init__(self, orders_str):
        self.orders_str = orders_str
        self.bidList = [] # max heap - buy list
        self.askList = [] # min heap - sell list

    # def PritList(self):
    # 	print('Bid/Buy List')
    # 	for order in self.bidList:
    # 		print(f"{order.orderNo} | {order.price} | {order.quantity}")
    # 	print("\n\nAsk/Sell List")
    # 	for order in self.askList:
    # 		print(f"{order.orderNo} | {order.price} | {order.quantity}")


    def BuildOrderBook(self):
        for orders in self.orders_str.split('\n'):
            orders_list = orders.split(' ')
            if len(orders_list) == 6:
                price = float(orders_list[4])
                quantity = int(orders_list[5])
                orderNo = orders_list[0]
                if orders_list[3] == 'buy':
                    order = Orders(-1.0 * price, orderNo, quantity)
                    heapq.heappush(self.bidList, order)
                    # heapq.heapify(self.bidList)
                elif orders_list[3] == 'sell':
                    order = Orders(price, orderNo, quantity)
                    heapq.heappush(self.askList, order)
                    # heapq.heapify(self.askList)
        heapq.heapify(self.bidList)
        heapq.heapify(self.askList)
        # self.PritList()

    def ExecuteOrder(self,executor):
        executor.Execute(self.bidList,self.askList)
    
class Executor(object):
    @staticmethod
    def Execute(bidList,askList):
        for bid in bidList:
            sellList = []
            bidPrice = -1.0 * bid.price
            for ask in askList:
                if bidPrice >= ask.price:
                    spread = bidPrice-ask.price
                    sellList.append((ask.orderNo,spread))
            sellList = sorted(sellList, key=lambda x: x[1])
            # print("="*100)
            # print(f"Bid : {bid.orderNo} {bidPrice} & Ask : {sellList}")
            # print("="*100)
            for sl in sellList:
                for ask in askList:
                    if sl[0]==ask.orderNo and ask.quantity > 0 and bid.quantity > 0:
                       tradeQunt = min(bid.quantity,ask.quantity) 
                       ask.quantity-=tradeQunt
                       bid.quantity-=tradeQunt
                       # print(f"Bid : {bid.orderNo} {bidPrice} {bid.quantity}")
                       # print(f"Ask : {ask.orderNo} {ask.price} {ask.quantity}")
                       print(f"{bid.orderNo} | {ask.price} | {tradeQunt} | {ask.orderNo}")


orders_str = """
#1 9:45 BAC sell 240.12 100
#2 9:46 BAC sell 237.45 90
#3 9:47 BAC buy 238.10 110
#4 9:48 BAC buy 237.80 10
#5 9:49 BAC buy 237.80 40
#6 9:50 BAC sell 236.00 50
"""

orderbook = OrderBook(orders_str)
orderbook.BuildOrderBook()
print("\n\nExecuted Orders\n\n")
orderbook.ExecuteOrder(Executor())

# 3 237.45 90 #2
# 3 236.00 20 #6
# 4 236.00 10 #6
# 5 236.00 20 #6