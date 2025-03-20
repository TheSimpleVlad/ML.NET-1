using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
    class Program
    {
        public class YNPhrase
        {
            [LoadColumn(0)]
            public string Phrase { get; set; }

            [LoadColumn(1)]
            public int Status { get; set; }
        }

        public class YNResult
        {
            [ColumnName("PredictedLabel")]
            public int Status { get; set; }

            [ColumnName("Score")]
            public float[] Scores { get; set; }
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 48);

            try
            {
                Console.WriteLine("[SYS] Загружаем данные");

                var data = new[]
                {
    // Примеры для класса "Нет" (0)
    new YNPhrase { Phrase = "Ебешь", Status = 0 },
    new YNPhrase { Phrase = "0 процентов %", Status = 0 },
    new YNPhrase { Phrase = "0%", Status = 0 },
    new YNPhrase { Phrase = "Я уверен, что это ложь", Status = 0 },
    new YNPhrase { Phrase = "Я уверен, что нет", Status = 0 },
    new YNPhrase { Phrase = "Нет", Status = 0 },
    new YNPhrase { Phrase = "Нет, это совсем неправильно.", Status = 0 },
    new YNPhrase { Phrase = "Абсолютно неверно.", Status = 0 },
    new YNPhrase { Phrase = "Это ошибка.", Status = 0 },
    new YNPhrase { Phrase = "Совершенно не так.", Status = 0 },
    new YNPhrase { Phrase = "Нет, это не подходит.", Status = 0 },
    new YNPhrase { Phrase = "Это ложное утверждение.", Status = 0 },
    new YNPhrase { Phrase = "Совершенно неправда.", Status = 0 },
    new YNPhrase { Phrase = "Я категорически не согласен.", Status = 0 },
    new YNPhrase { Phrase = "Нет, это невозможно.", Status = 0 },
    new YNPhrase { Phrase = "Это неверный ответ.", Status = 0 },
    new YNPhrase { Phrase = "Это звучит абсолютно неправильно.", Status = 0 },
    new YNPhrase { Phrase = "Ложь", Status = 0 },
    new YNPhrase { Phrase = "Это не то, что нужно.", Status = 0 },
    new YNPhrase { Phrase = "Пиздишь!", Status = 0 },
    new YNPhrase { Phrase = "Врешь", Status = 0 },

    // Примеры для класса "Да" (1)
    new YNPhrase { Phrase = "Истина", Status = 1 },
    new YNPhrase { Phrase = "Правда", Status = 1 },
    new YNPhrase { Phrase = "Не ебешь", Status = 1 },
    new YNPhrase { Phrase = "100 процентов %", Status = 1 },
    new YNPhrase { Phrase = "Это не ложь", Status = 1 },
    new YNPhrase { Phrase = "Конечно!", Status = 1 },
    new YNPhrase { Phrase = "Я уверен", Status = 1 },
    new YNPhrase { Phrase = "Точно", Status = 1 },
    new YNPhrase { Phrase = "Да", Status = 1 },
    new YNPhrase { Phrase = "Да, всё верно!", Status = 1 },
    new YNPhrase { Phrase = "Это абсолютно правильно.", Status = 1 },
    new YNPhrase { Phrase = "Я согласен.", Status = 1 },
    new YNPhrase { Phrase = "Да, это так.", Status = 1 },
    new YNPhrase { Phrase = "Совершенно верно.", Status = 1 },
    new YNPhrase { Phrase = "Это правильный ответ.", Status = 1 },
    new YNPhrase { Phrase = "Абсолютно точно.", Status = 1 },
    new YNPhrase { Phrase = "Вы правы.", Status = 1 },
    new YNPhrase { Phrase = "Да, это то, что нужно.", Status = 1 },
    new YNPhrase { Phrase = "Безусловно, это так.", Status = 1 },
    new YNPhrase { Phrase = "Это звучит как истина.", Status = 1 },
    new YNPhrase { Phrase = "Вы абсолютно правы.", Status = 1 },
    new YNPhrase { Phrase = "Не пиздишь", Status = 1 },
    new YNPhrase { Phrase = "Не врешь", Status = 1 },
    new YNPhrase { Phrase = "Агась", Status = 1 },

    // Примеры для класса "Возможно" (2)
    new YNPhrase { Phrase = "50 процентов %", Status = 2 },
    new YNPhrase { Phrase = "Я не знаю", Status = 2 },
    new YNPhrase { Phrase = "Я не уверен", Status = 2 },
    new YNPhrase { Phrase = "Возможно", Status = 2 },
    new YNPhrase { Phrase = "Трудно сказать.", Status = 2 },
    new YNPhrase { Phrase = "Это неоднозначный ответ.", Status = 2 },
    new YNPhrase { Phrase = "Возможно, так и есть.", Status = 2 },
    new YNPhrase { Phrase = "Не могу с уверенностью ответить.", Status = 2 },
    new YNPhrase { Phrase = "Это вопрос интерпретации.", Status = 2 },
    new YNPhrase { Phrase = "Может быть.", Status = 2 },
    new YNPhrase { Phrase = "Я не уверен.", Status = 2 },
    new YNPhrase { Phrase = "Это зависит от контекста.", Status = 2 },
    new YNPhrase { Phrase = "Может, но не точно.", Status = 2 },
    new YNPhrase { Phrase = "Тут есть сомнения.", Status = 2 },
    new YNPhrase { Phrase = "Не ясно, правда это или нет.", Status = 2 },
    new YNPhrase { Phrase = "Ответ неоднозначен.", Status = 2 },

    // Примеры для класса "Неопределимый" (3)
    new YNPhrase { Phrase = "ЦУОТ", Status = 3 },
    new YNPhrase { Phrase = "СОЛ", Status = 3 },
    new YNPhrase { Phrase = "СИЛ", Status = 3 },
    new YNPhrase { Phrase = "Вы пр", Status = 3 },
    new YNPhrase { Phrase = "Это правда?", Status = 3 },
    new YNPhrase { Phrase = "Это точно ложь?", Status = 3 },
    new YNPhrase { Phrase = "Что-нибудь", Status = 3 },
    new YNPhrase { Phrase = "Что-то", Status = 3 },
    new YNPhrase { Phrase = "Я уверен, что это прекрасно", Status = 3 },
    new YNPhrase { Phrase = "Я уверен, что он прекрасен", Status = 3 },
    new YNPhrase { Phrase = "Я прекрасен", Status = 3 },
    new YNPhrase { Phrase = "Я очень добр", Status = 3 },
    new YNPhrase { Phrase = "Какой прекрасный закат!", Status = 3 },
    new YNPhrase { Phrase = "Это очень любопытно.", Status = 3 },
    new YNPhrase { Phrase = "Что вы думаете об этой книге?", Status = 3 },
    new YNPhrase { Phrase = "Привет, давно не виделись!", Status = 3 },
    new YNPhrase { Phrase = "Сегодня отличный день.", Status = 3 },
    new YNPhrase { Phrase = "Это замечательный фильм.", Status = 3 },
    new YNPhrase { Phrase = "Почему небо синее?", Status = 3 },
    new YNPhrase { Phrase = "Мне нравится вкус этого блюда.", Status = 3 },
    new YNPhrase { Phrase = "Это место такое уютное.", Status = 3 },
    new YNPhrase { Phrase = "Как давно вы этим занимаетесь?", Status = 3 },
    new YNPhrase { Phrase = "Я об этом раньше не задумывался.", Status = 3 },
    new YNPhrase { Phrase = "Какая у вас любимая книга?", Status = 3 },
    new YNPhrase { Phrase = "Пиздец.", Status = 3 },
    new YNPhrase { Phrase = "Я бы хотел узнать больше.", Status = 3 }
};
                var trainData = mlContext.Data.LoadFromEnumerable(data);

                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Status") // Преобразуем Status в Label
    .Append(mlContext.Transforms.Text.NormalizeText("Normal", "Phrase"))
    .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Normal"))
    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
        labelColumnName: "Label",
        featureColumnName: "Features",
        maximumNumberOfIterations: 200000)) // Тренируем модель
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "Label")); // Преобразуем ключ обратно в значение
                var preview = pipeline.Preview(trainData);
                foreach (var row in preview.RowView)
                {
                    foreach (var column in row.Values)
                    {
                        Console.WriteLine($"{column.Key} --> {column.Value}");
                    }
                }
                Console.WriteLine("[SYS] Обучаем модель");
                var model = pipeline.Fit(trainData);
                Console.WriteLine("[SYS] Вычисляем");
                var predictionEngine = mlContext.Model.CreatePredictionEngine<YNPhrase, YNResult>(model);
                Console.Clear();
                while (true)
                {
                    var v = new YNPhrase { Phrase = Console.ReadLine() };
                    var prediction = predictionEngine.Predict(v);

                    int predictedIndex = Array.IndexOf(prediction.Scores, prediction.Scores.Max());
                    string result = "Error / Ошибка";
                    if (predictedIndex == 0)
                        result = "No / Нет";
                    else if (predictedIndex == 1)
                        result = "Yes / Да";
                    else if (predictedIndex == 2)
                        result = "Intermidiate / Возможно";
                    else if (predictedIndex == 3)
                        result = "Unknown / Неопределимый";
                    if (predictedIndex == 0)
                        Console.ForegroundColor = ConsoleColor.Red;
                    else if (predictedIndex == 1)
                        Console.ForegroundColor = ConsoleColor.Green;
                    else if (predictedIndex == 2)
                        Console.ForegroundColor = ConsoleColor.DarkYellow;
                    else if (predictedIndex == 3)
                        Console.ForegroundColor = ConsoleColor.Blue;
                    Console.CursorTop--;
                    Console.WriteLine($"{v.Phrase} --> {result} --> {prediction.Scores[predictedIndex] * 100}%");
                    Console.ForegroundColor = ConsoleColor.Gray;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("EXCEPTION --> " + e.Message);
                Console.WriteLine("CALLSTACK:\n" + e.StackTrace);
            }
        }
    }
}
