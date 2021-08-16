{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled3.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyO4ggPPCDBVvkf4zoaXNqjK",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/cindyhfls/NMA_DL_2021_project/blob/main/Pearson_correlation.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "w7ZAWddz_aUG"
      },
      "source": [
        "def pearson_corr_tensor(input, output):\n",
        "  rpred = output.detach().cpu().numpy()\n",
        "  rreal = input.detach().cpu().numpy()\n",
        "  rpred_flat = np.ndarray.flatten(rpred)\n",
        "  rreal_flat = np.ndarray.flatten(rreal)\n",
        "  corrcoeff = np.corrcoef(rpred_flat, rreal_flat)\n",
        "  return corrcoeff[0,1]"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}