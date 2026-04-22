// See https://aka.ms/new-console-template for more information

using DotNetRag01;
using Microsoft.ML;

// 2 - Load the data

var data = new List<StudentData>
{
    new() { StudyHours = 4, PreviousGrade = 4.0f, FinalGrade = 5.5f },
    new() { StudyHours = 4, PreviousGrade = 5.5f, FinalGrade = 6.5f },
    new() { StudyHours = 6, PreviousGrade = 6.5f, FinalGrade = 7.0f },
    new() { StudyHours = 8, PreviousGrade = 7.5f, FinalGrade = 8.0f },
    new() { StudyHours = 10, PreviousGrade = 8.5f, FinalGrade = 9.0f },
};

var context = new MLContext();
var dataView = context.Data.LoadFromEnumerable(data);

// 3 - Build the model

var pipeline = context.Transforms
    .Concatenate("Features", inputColumnNames: ["StudyHours", "PreviousGrade"])
    .Append(context.Regression.Trainers.Sdca(
        labelColumnName: "FinalGrade",
        maximumNumberOfIterations: 100));

var model = pipeline.Fit(dataView);

// 4 - Make predictions

var predictionEngine = context.Model
    .CreatePredictionEngine<StudentData, StudentPrediction>(model);

var prediction = predictionEngine.Predict(new StudentData
{
    StudyHours = 7,
    PreviousGrade = 7.0f
});
    
// 5. Running the model

Console.WriteLine($"Predicted final grade: {prediction.Score:F1}");