class User:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class Split:
    def __init__(self, user):
        self.user = user
        self.amount = None

    def get_user(self):
        return self.user

    def set_user(self, user):
        self.user = user

    def get_amount(self):
        return self.amount

    def set_amount(self, amount):
        self.amount = amount


class PercentSplit(Split):
    def __init__(self, user, percent):
        super().__init__(user)
        self.percent = percent

    def get_percent(self):
        return self.percent

    def set_percent(self, percent):
        self.percent = percent


class EqualSplit(Split):
    pass


class ExactSplit(Split):
    def __init__(self, user, amount):
        super().__init__(user)
        self.amount = amount

    def get_amount(self):
        return self.amount

    def set_amount(self, amount):
        self.amount = amount


class ExpenseMetadata:
    def __init__(self, name, notes):
        self.name = name
        self.notes = notes

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_notes(self):
        return self.notes

    def set_notes(self, notes):
        self.notes = notes


class Expense:
    def __init__(self, amount, paid_by, splits, metadata):
        self.amount = amount
        self.paid_by = paid_by
        self.splits = splits
        self.metadata = metadata

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def get_amount(self):
        return self.amount

    def set_amount(self, amount):
        self.amount = amount

    def get_paid_by(self):
        return self.paid_by

    def set_paid_by(self, paid_by):
        self.paid_by = paid_by

    def get_splits(self):
        return self.splits

    def set_splits(self, splits):
        self.splits = splits

    def get_metadata(self):
        return self.metadata

    def set_metadata(self, metadata):
        self.metadata = metadata

    def validate(self):
        pass


class EqualExpense(Expense):
    def validate(self):
        for split in self.get_splits():
            if not isinstance(split, EqualSplit):
                return False
        return True


class ExactExpense(Expense):
    def validate(self):
        for split in self.get_splits():
            if not isinstance(split, ExactSplit):
                return False

        total_amount = self.get_amount()
        sum_split_amount = 0
        for split in self.get_splits():
            exact_split = split
            sum_split_amount += exact_split.get_amount()

        if total_amount != sum_split_amount:
            return False

        return True


class PercentExpense(Expense):
    def validate(self):
        for split in self.get_splits():
            if not isinstance(split, PercentSplit):
                return False

        total_percent = 100
        sum_split_percent = 0
        for split in self.get_splits():
            exact_split = split
            sum_split_percent += exact_split.get_percent()

        if total_percent != sum_split_percent:
            return False

        return True


class ExpenseType:
    EQUAL = "EQUAL"
    EXACT = "EXACT"
    PERCENT = "PERCENT"


class ExpenseService:
    @staticmethod
    def create_expense(expense_type, amount, paid_by, splits, metadata):
        if expense_type == ExpenseType.EXACT:
            return ExactExpense(amount, paid_by, splits, metadata)
        elif expense_type == ExpenseType.PERCENT:
            for split in splits:
                percent_split = split
                split.set_amount((amount * percent_split.get_percent()) / 100.0)
            return PercentExpense(amount, paid_by, splits, metadata)
        elif expense_type == ExpenseType.EQUAL:
            total_splits = len(splits)
            split_amount = round(amount * 100 / total_splits) / 100.0
            for split in splits:
                split.set_amount(split_amount)
            splits[0].set_amount(split_amount + (amount - split_amount * total_splits))
            return EqualExpense(amount, paid_by, splits, metadata)
        else:
            return None


class ExpenseManager:
    def __init__(self):
        self.expenses = []
        self.user_map = {}
        self.balance_sheet = {}

    def add_user(self, user):
        self.user_map[user.get_id()] = user
        self.balance_sheet[user.get_id()] = {}

    def add_expense(self, expense_type, amount, paid_by, splits, metadata):
        expense = ExpenseService.create_expense(expense_type, amount, self.user_map[paid_by], splits, metadata)
        self.expenses.append(expense)
        for split in expense.get_splits():
            paid_to = split.get_user().get_id()
            # conditional assignment
            balances = self.balance_sheet[paid_by]
            if paid_to not in balances:
                balances[paid_to] = 0.0
            balances[paid_to] += split.get_amount()
            # balance update
            balances = self.balance_sheet[paid_to]
            if paid_by not in balances:
                balances[paid_by] = 0.0
            balances[paid_by] -= split.get_amount()

    def show_balance(self, user_id):
        is_empty = True
        for user, balance in self.balance_sheet[user_id].items():
            if balance != 0:
                self.print_balance(user_id, user, balance)
                is_empty = False

        if is_empty:
            print("No balances")

    def show_balances(self):
        is_empty = True
        for user1, user_balances in self.balance_sheet.items():
            for user2, balance in user_balances.items():
                if balance > 0:
                    self.print_balance(user1, user2, balance)
                    is_empty = False

        if is_empty:
            print("No balances")

    def print_balance(self, user1, user2, amount):
        user1_name = self.user_map[user1].get_name()
        user2_name = self.user_map[user2].get_name()
        if amount < 0:
            print(user1_name + " owes " + user2_name + ": " + str(abs(amount)))
        elif amount > 0:
            print(user2_name + " owes " + user1_name + ": " + str(abs(amount)))


if __name__ == "__main__":
    expense_manager = ExpenseManager()

    expense_manager.add_user(User("u1", "User1"))
    expense_manager.add_user(User("u2", "User2"))
    expense_manager.add_user(User("u3", "User3"))
    expense_manager.add_user(User("u4", "User4"))

    while True:
        command = input()
        commands = command.split(" ")
        command_type = commands[0]

        if command_type == "SHOW":
            if len(commands) == 1:
                expense_manager.show_balances()
            else:
                expense_manager.show_balance(commands[1])
        elif command_type == "EXPENSE":
            paid_by = commands[1]
            amount = float(commands[2])
            no_of_users = int(commands[3])
            expense_type = commands[4 + no_of_users]
            splits = []
            for i in range(no_of_users):
                user_id = commands[4 + i]
                if expense_type == "EQUAL":
                    splits.append(EqualSplit(expense_manager.user_map[user_id]))
                elif expense_type == "EXACT":
                    splits.append(ExactSplit(expense_manager.user_map[user_id], float(commands[5 + no_of_users + i])))
                elif expense_type == "PERCENT":
                    splits.append(PercentSplit(expense_manager.user_map[user_id], float(commands[5 + no_of_users + i])))
            expense_manager.add_expense(expense_type, amount, paid_by, splits, None)



# SHOW
# SHOW u1
# EXPENSE u1 1000 4 u1 u2 u3 u4 EQUAL
# SHOW u4
# SHOW u1
# EXPENSE u1 1250 2 u2 u3 EXACT 370 880
# SHOW
# EXPENSE u4 1200 4 u1 u2 u3 u4 PERCENT 40 20 20 20
# SHOW u1
# SHOW