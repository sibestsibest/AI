using Ai.Phase8.Demos;

// ======================================================================
// MINI AI — PHASE 8: MINI TRANSFORMER
//
//     Embedding -> Positional Encoding -> Self Attention -> Feed Forward
//     -> Layer Normalization -> Output
//
// Đầy đủ các thành phần kiến trúc của GPT, thu nhỏ lại. Backpropagation
// chạy xuyên qua attention và layer norm, có gradient check xác nhận.
// ======================================================================

Console.WriteLine("############################################");
Console.WriteLine("#   MINI AI — PHASE 8: MINI TRANSFORMER    #");
Console.WriteLine("############################################\n");

TransformerDemo.Run();

Console.WriteLine("\n\n=== PHASE 8 HOÀN TẤT ===");
