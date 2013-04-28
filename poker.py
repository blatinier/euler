cards = {'2': 2,
         '3': 3,
         '4': 4,
         '5': 5,
         '6': 6,
         '7': 7,
         '8': 8,
         '9': 9,
         'T': 10,
         'J': 11,
         'Q': 12,
         'K': 13,
         'A': 14}


class PokerHand:
    def __init__(self, poker_str_l):
        self.hand = []
        self.card_values = []
        self.card_suits = []
        for card in poker_str_l:
            self.hand.append((cards[card[0]], card[1]))
            self.card_values.append(cards[card[0]])
            self.card_suits.append(card[1])
        self.card_values = sorted(self.card_values)
        self.same_suit = all(self.card_suits[0] == cs for cs in self.card_suits)
        self.consecutive = self.card_values == [self.card_values[0] + i for i in range(5)]
        self.values_set = set(self.card_values)
        self.card_values_count = dict(zip(self.card_values, map(self.card_values.count, self.card_values)))
        self.card_values_count_val = sorted(self.card_values_count.values())
        self.max_card_val_count = max(self.card_values_count_val)

    def has_royal_flush(self):
        return self.same_suit and self.card_values[0] == 10

    def straight_flush(self):
        return self.same_suit and self.consecutive

    def is_four_of_a_kind(self):
        return self.max_card_val_count == 4

    def is_full(self):
        return self.card_values_count_val == [2,3]

    def is_flush(self):
        return self.same_suit

    def is_straight(self):
        return self.consecutive

    def if_three_of_kind(self):
        return self.max_card_val_count == 3
        
    def is_two_pair(self):
        return self.card_values_count_val == [1,2,2]

    def if_pair(self):
        return self.max_card_val_count == 2

    def top_card(self):
        return max(self.card_values)

    def __ge__(self, other):
        if self.has_royal_flush():
            print "Win by Royal flush"
            return True
        elif self.straight_flush():
            if not other.has_royal_flush() and (not other.straight_flush() \
            or (other.straight_flush() and self.top_card() > other.top_card())):
                print "Win by strait flush"
                return True
            else:
                return False
        elif self.is_four_of_a_kind():
            if other.has_royal_flush() or other.straight_flush():
                return False
            if not other.is_four_of_a_kind():
                print "Win by four a kind easy"
                return True
            for k, v in self.card_values_count.items():
                if v == 4:
                    s_card = k
                else:
                    s_ocard = k
            for k, v in other.card_values_count.items():
                if v == 4:
                    o_card = k
                else:
                    o_ocard = k
            if s_card > o_card:
                print "Win by four a kind high 4"
                return True
            elif s_card == o_card:
                if s_ocard > o_ocard:
                    print "Win by four a kind other higher"
                return s_ocard > o_ocard
            else:
                return False
        elif self.is_full():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind():
                return False
            if not other.is_full():
                print "Win by full"
                return True
            for k, v in self.card_values_count.items():
                if v == 3:
                    s_card1 = k
                if v == 2:
                    s_card2 = k
            for k, v in other.card_values_count.items():
                if v == 3:
                    o_card1 = k
                if v == 2:
                    o_card2 = k
            if s_card1 > o_card1:
                print "Win by full better 3"
                return True
            elif s_card1 == o_card1:
                if s_card2 > o_card2:
                    print "Win by full better 2"
                    return True
            return False
        elif self.is_flush():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full():
                return False
            if not other.is_flush():
                print "Win by flush"
                return True
            return self.top_card() > other.top_card()
        elif self.is_straight():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full() \
                or other.is_flush():
                return False
            if not other.is_straight():
                print "Win by strait"
                return True
            return self.top_card() > other.top_card()
        elif self.if_three_of_kind():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full() \
                or other.is_flush() or other.is_straight():
                return False
            if not other.if_three_of_kind():
                print "Win by tierce"
                return True
            s_ocard = []
            o_ocard = []
            for k, v in self.card_values_count.items():
                if v == 3:
                    s_card = k
                else:
                    s_ocard.append(k)
            for k, v in other.card_values_count.items():
                if v == 3:
                    o_card = k
                else:
                    o_ocard.append(k)
            if s_card > o_card:
                print "Win by tierce better"
                return True
            elif s_card == o_card:
                s_ocard = sorted(s_ocard)
                o_ocard = sorted(o_ocard)
                if s_ocard[1] > o_ocard[1]:
                    print "Win by tierce 1 better"
                    return True
                elif s_ocard[1] == o_ocard[1]:
                    if s_ocard[0] > o_ocard[0]:
                        print "Win by tierce 2 better"
                    return s_ocard[0] > o_ocard[0]
            return False
        elif self.is_two_pair():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full() \
                or other.is_flush() or other.is_straight() \
                or other.if_three_of_kind():
                return False
            if not other.is_two_pair():
                print "Win by two pairs"
                return True
            s_pcard = []
            o_pcard = []
            for k, v in self.card_values_count.items():
                if v == 1:
                    s_ocard = k
                else:
                    s_pcard.append(k)
            for k, v in other.card_values_count.items():
                if v == 1:
                    o_ocard = k
                else:
                    o_pcard.append(k)
            s_pcard = sorted(s_pcard)
            o_pcard = sorted(o_pcard)
            if s_pcard[1] > o_pcard[1]:
                print "Win by two pairs first"
                return True
            elif s_pcard[1] == o_pcard[1]:
                if s_pcard[0] > o_pcard[0]:
                    print "Win by two pairs second"
                    return True
                elif s_pcard[0] == o_pcard[0]:
                    if s_ocard > o_ocard:
                        print "Win by two pairs third"
                    return s_ocard > o_ocard
            return False
        elif self.if_pair():
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full() \
                or other.is_flush() or other.is_straight() \
                or other.if_three_of_kind() or other.is_two_pair():
                return False
            if not other.if_pair():
                print "Win by pair"
                return True
            s_ocard = []
            o_ocard = []
            for k, v in self.card_values_count.items():
                if v == 2:
                    s_card = k
                else:
                    s_ocard.append(k)
            for k, v in other.card_values_count.items():
                if v == 2:
                    o_card = k
                else:
                    o_ocard.append(k)
            if s_card > o_card:
                print "Win by pair better"
                return True
            elif s_card == o_card:
                s_ocard = sorted(s_ocard)
                o_ocard = sorted(o_ocard)
                if s_ocard[2] > o_ocard[2]:
                    print "Win by pair better 1"
                    return True
                elif s_ocard[2] == o_ocard[2]:
                    if s_ocard[1] > o_ocard[1]:
                        print "Win by pair better 2"
                        return True
                    elif s_ocard[1] == o_ocard[1]:
                        if s_ocard[0] > o_ocard[0]:
                            print "Win by pair better 3"
                        return s_ocard[0] > o_ocard[0]
            return False
        else:
            if other.has_royal_flush() or other.straight_flush() \
                or other.is_four_of_a_kind() or other.is_full() \
                or other.is_flush() or other.is_straight() \
                or other.if_three_of_kind() or other.is_two_pair() \
                or other.if_pair():
                return False
            if self.top_card() > other.top_card():
                print "Win by top"
            return self.top_card() > other.top_card()
        print "else ??"
        return False

print "examples"
example = ['5H 5C 6S 7S KD 2C 3S 8S 8D TD', # Player 2
'5D 8C 9S JS AC 2C 5C 7D 8S QH', # Player 1
'2D 9C AS AH AC 3D 6D 7D TD QD', # Player 2
'4D 6S 9H QH QC 3D 6D 7H QD QS', # Player 1
'2H 2D 4C 4D 4S 3C 3D 3S 9S 9D'] # Player 1
c = 0
f = open('poker.txt')
for l in f.readlines():
#for l in example:
    hands = l.strip().split(' ')
    p1, p2 = PokerHand(hands[0:5]), PokerHand(hands[5:10])
    if p1.__ge__(p2):
        print "Player 1", l
        c += 1
    elif p2.__ge__(p1):
        print "Player 2", l
print c
