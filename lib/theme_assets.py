from matplotlib.colors import LinearSegmentedColormap

CGREEN = "rgb(15, 157, 88)"
CRED = "rgb(219, 68, 55)"
CBLUE = "rgb(66, 133, 244)"
CYELLOW = "rgb(244, 160, 0)"
CORANGE = "rgb(255, 87, 34)"
CCYAN = "rgb(0, 188, 212)"
CWHITE = "rgb(255, 255, 255)"


def C(i):
    return {
        0: "#4285F4",
        1: "#FF6D01",
        2: "#46BDC6",
        3: "#F4B400",
        # 1: '#DB4437',
        # 3: '#0F9D58',
    }[i % 4]


def DC(i):
    return {
        0: "#2066a8",
        1: "#8ec1da",
        2: "#cde1ec",
        3: "#ededed",
        4: "#f6d6c2",
        5: "#d47264",
        6: "#ae282c",
    }[i % 7]


CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#DB4437", "#0F9D58"], N=7
)
