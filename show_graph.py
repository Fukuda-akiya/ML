import matplotlib.pyplot as plt

def show_graph(X, Y, x, y):
    fig, ax = plt.subplots(dpi=100)
    ax.scatter(X, Y, marker='.')
    ax.plot(x, y, "tab:red")
    #ax.set_title("最高気温とアイスクリーム・シャーベットの支出額")
    #ax.set_xlabel("最高気温の月平均(度)")
    #ax.set_ylabel("支出額（円）")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid()
    plt.show()
