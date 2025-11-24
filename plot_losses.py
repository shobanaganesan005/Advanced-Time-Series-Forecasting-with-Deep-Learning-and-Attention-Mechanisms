
import json
import matplotlib.pyplot as plt


def main():
    with open("results.json") as f:
        results = json.load(f)

    for name, content in results.items():
        history = content["history"]
        plt.figure()
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss curves - {name}")
        plt.legend()
        plt.savefig(f"loss_{name}.png")
        print(f"Saved loss_{name}.png")


if __name__ == "__main__":
    main()
