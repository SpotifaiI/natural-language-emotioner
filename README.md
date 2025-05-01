# Natural Language Emotioner
Application to recognize emotion in random comments sent to API for Machine Learning class.

# Equipe

* Cristian Prochnow
* Gustavo Henrique Dias
* Lucas Willian de Souza Serpa
* Marlon de Souza
* Ryan Gabriel Mazzei Bromati

# Getting Started

Para rodar o projeto, basta executar os comandos abaixo.

```shell
# usando Docker
$ docker compose up



# ou se quiser rodar os pacotes manualmente
$ pip install -r requirements.txt
$ fastapi run app.py
```

As rotas para uso estão todas no arquivo [de requisições](./requests.http).

A API é formada por duas rotas principais (além do índice), sendo então a rota de `/ask`, ao qual é necessário enviar um JSON no formato abaixo.

```json
{
  "comment": "conteúdo bom demais"
}
```

Recebendo então, como resposta, o sentimento representado por aquele comentário que foi enviado.

```json
{
  "success": true,
  "emotion": "alegria",
  "comment": "conteúdo bom demais"
}
```

E, para análise, há a rota `/stats`, que fica responsável por realizar todos os cálculos referentes às métricas de algoritmos de classificação, para avaliarmos melhor o processamento Nayve Bayes.

```json
{
  "success": true,
  "metrics": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1_score": 1.0,
    "confusion_matrix": [
      [
        1,
        0
      ],
      [
        0,
        2
      ]
    ],
    "classification_report": {
      "alegria": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 1.0
      },
      "desgosto": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 2.0
      },
      "accuracy": 1.0,
      "macro avg": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 3.0
      },
      "weighted avg": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 3.0
      }
    }
  }
}
```

Exemplos podem ser encontrados diretamente no arquivo de requisições citado anteriormente.
