clc; clear;
pkg load video;
pkg load image;

% Cargar un video en un objeto
video_read = VideoReader('original.mp4'); % Comando para cargar un video

fr = 1024; % Numero de fotogramas
%fr = video_read.NumberOfFrames; % Numero de fotogramas
m = video_read.Height; % Numero de filas de cada fotograma
n = video_read.Width; % Numero de columnas de cada fotograma

fr
m
n

% Crear el nuevo video
video_write = VideoWriter('noise.mp4');
% Abrir el archivo del video nuevo
open(video_write);

% Procesar los fotogramas del video
disp('Processing...');
for k = 1:fr
  k
  % Leer el fotograma
  frame = readFrame(video_read);
  frame = frame(:,:,1);
  % Agregar ruido al fotograma
  frame = imnoise(frame,'gaussian',0.1);
  % Escribir cada fotograma del video
  writeVideo(video_write, frame);
end

close(video_write);
disp('Finished');
