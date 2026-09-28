"""Classify messages with the saved model.

    python predict.py                       # run built-in demo messages
    python predict.py "your message here"   # classify your own message(s)
"""
import sys

from src.inference import classify

DEMO = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT! Your account has been suspended. Verify your details immediately.",
    "Can you send me the notes from today's class?",
    "FREE entry into our weekly draw for a brand new iPhone! Text WIN to 80086 now!",
    "Reminder: your electricity bill payment is due tomorrow, please pay to avoid late fee.",
    "fr33 msg: y0u w1n a prize, txt CLAIM to 81010",
    "Mom, call me back when you get a chance",
]

if __name__ == "__main__":
    msgs = sys.argv[1:] or DEMO
    for r in classify(msgs):
        print(f"[{r['decision']:6}] p(spam)={r['spam_probability']:.3f}  {r['message']}")
