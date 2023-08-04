package com.programminghut.realtime_object

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.core.content.ContextCompat
import com.programminghut.realtime_object.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Основной класс MainActivity, наследуется от AppCompatActivity
class MainActivity : AppCompatActivity() {

    lateinit var labels: List<String>    // Переменная для хранения меток (названий) объектов
    var colors =
        listOf<Int>(           // Список цветов для обозначения рамок вокруг распознанных объектов
            Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
            Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED
        )
    val paint =
        Paint()                 // Экземпляр класса Paint для рисования на холсте (например, рамок)
    lateinit var imageProcessor: ImageProcessor // Обработчик изображения, который будет использоваться для изменения изображения перед передачей в модель
    lateinit var bitmap: Bitmap          // Bitmap для обработки изображения
    lateinit var imageView: ImageView   // Элемент пользовательского интерфейса для отображения изображения
    lateinit var cameraDevice: CameraDevice // Экземпляр класса CameraDevice для работы с камерой устройства
    lateinit var handler: Handler       // Обработчик (Handler) для выполнения задач в фоновом потоке
    lateinit var cameraManager: CameraManager // Менеджер камеры для взаимодействия с камерой устройства
    lateinit var textureView: TextureView  // TextureView для предпросмотра изображения с камеры
    lateinit var model: SsdMobilenetV11Metadata1 // Модель для распознавания объектов (предположительно, SsdMobilenetV11Metadata1)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Устанавливаем макет для активности из ресурсов (activity_main.xml)
        setContentView(R.layout.activity_main)

        // Запрашиваем разрешение на использование камеры (предположительно)
        get_permission()

        // Загружаем метки (названия объектов) из файла "labels.txt"
        labels = FileUtil.loadLabels(this, "labels.txt")

        // Создаем ImageProcessor для обработки изображения, в данном случае изменяем размер изображения до 300x300
        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()

        // Инициализируем модель для распознавания объектов (SsdMobilenetV11Metadata1)
        model =
            SsdMobilenetV11Metadata1.newInstance(this) // this предоставляет доступ к ресурсам и функциям Android-приложения.

        // Создаем и запускаем фоновый поток (HandlerThread) для обработки видеопотока
        val handlerThread =
            HandlerThread("videoThread") // Фоновый поток - это поток, который работает параллельно с основным потоком, и может выполнять долгие операции без замедления пользовательского интерфейса.
        handlerThread.start() // фоновый поток запускается с помощью метода start(). После вызова этого метода, фоновый поток начнет свою работу и будет выполнять операции внутри него.
        handler =
            Handler(handlerThread.looper) // Handler позволяет отправлять задачи на выполнение в фоновый поток через его цикл обработки сообщений.

        // Находим ImageView по идентификатору и связываем с переменной imageView
        imageView = findViewById(R.id.imageView)

        // Находим TextureView по идентификатору и связываем с переменной textureView
        textureView =
            findViewById(R.id.textureView) // TextureView представляет собой компонент интерфейса, который может использоваться для отображения видео и графики,
        // а также для предпросмотра изображения с камеры в реальном времени.

        // Устанавливаем слушатель текстурной поверхности (SurfaceTextureListener) для TextureView
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {

            // Метод вызывается, когда текстурная поверхность готова к использованию
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera() // Открываем камеру
            }

            // Метод вызывается, когда размер текстурной поверхности изменяется
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            // Метод вызывается, когда текстурная поверхность уничтожается
            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            // Метод вызывается, когда текстурная поверхность обновляется
            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                // Получаем из TextureView текущее изображение как Bitmap
                bitmap = textureView.bitmap!!

                // Преобразуем изображение в TensorImage
                var image = TensorImage.fromBitmap(bitmap)

                // Производим предварительную обработку изображения с использованием ImageProcessor
                image = imageProcessor.process(image)

                // Подаем обработанное изображение на модель для получения результатов
                val outputs = model.process(image)

                // Извлекаем результаты (локации, классы, оценки и количество обнаружений) из выводов модели
                val locations =
                    outputs.locationsAsTensorBuffer.floatArray //  представляет выходные данные модели, содержащие координаты (локации) рамок вокруг обнаруженных объектов.
                val classes =
                    outputs.classesAsTensorBuffer.floatArray // представляет выходные данные модели, содержащие классы (идентификаторы) обнаруженных объектов.
                val scores =
                    outputs.scoresAsTensorBuffer.floatArray // выходные данные модели, содержащие оценки (вероятности) обнаруженных объектов.
                val numberOfDetections =
                    outputs.numberOfDetectionsAsTensorBuffer.floatArray // количество обнаруженных объектов.

                // Создаем копию изображения (Bitmap), с которой будем работать
                var mutable = bitmap.copy(
                    Bitmap.Config.ARGB_8888,
                    true
                ) // создаем копию исходного изображения (bitmap) с определенными параметрами. Bitmap.Config.ARGB_8888 указывает на формат цветового пространства изображения (каждый пиксель представлен четырьмя каналами
                val canvas =
                    Canvas(mutable) // создаем объект Canvas, который предоставляет поверхность для рисования на битмапе

                // Получаем высоту и ширину изображения
                val h = mutable.height
                val w = mutable.width

                // Настройка параметров рисования (размер текста и толщина линий)
                paint.textSize =
                    h / 15f // h представляет высоту изображения, и эта строка кода устанавливает размер текста в 1/15 высоты изображения.
                // Это позволяет настроить размер текста в соответствии с размером изображения.
                paint.strokeWidth =
                    h / 85f // устанавливает толщину линии в 1/85 высоты изображения. Это позволяет настроить толщину рамок в соответствии с размером изображения.
                var x = 0 //  будет использоваться для вычисления индексов в массиве locations

                // Итерируемся по результатам распознавания
                // scores это список, содержащий оценки (вероятности) для каждого обнаруженного объекта на изображении.
                // Оценка показывает, насколько уверенна модель в том, что определенный объект присутствует на изображении.
                scores.forEachIndexed { index, fl -> // индекс, вероятность
                    x =
                        index // это переменная, используемая для вычисления индексов в массиве locations.
                    x *= 4 // для получения соответствующих координат в массиве locations, которые представляют локации (границы) обнаруженных объектов на изображении.

                    if (fl > 0.5) { //  это условие, которое проверяет, является ли оценка (вероятность) текущего объекта больше 0.5.
                        // Это условие фильтрует объекты с низкой уверенностью, и только объекты с вероятностью более 0.5 будут
                        // подвергаться дальнейшей обработке и рисованию рамок вокруг них.


                        // Устанавливаем цвет и стиль для рисования рамки вокруг объекта
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE

                        // Рисуем прямоугольник (рамку) вокруг объекта
                        canvas.drawRect(
                            RectF(
                                locations.get(x + 1) * w,
                                locations.get(x) * h,
                                locations.get(x + 3) * w,
                                locations.get(x + 2) * h
                            ), paint
                        )

                        // Меняем стиль на заливку для вывода названия объекта и оценки
                        paint.style = Paint.Style.FILL

                        // Рисуем текст с названием объекта и оценкой
                        canvas.drawText(
                            labels.get(
                                classes.get(index).toInt()
                            ) + " " + fl.toString(),
                            locations.get(x + 1) * w,
                            locations.get(x) * h,
                            paint
                        )
                    }
                }
                imageView.setImageBitmap(mutable)  // Устанавливаем обновленное изображение на ImageView
            }
        }
        cameraManager =
            getSystemService(Context.CAMERA_SERVICE) as CameraManager // Получаем экземпляр CameraManager для работы с камерой
    }

    // Метод вызывается при завершении активности
    override fun onDestroy() {
        super.onDestroy()
        model.close() // Закрываем модель для освобождения ресурсов
    }


    @SuppressLint("MissingPermission") // это аннотация, которая подавляет предупреждение о отсутствии разрешения на использование камеры.
    // В данном случае, предполагается, что разрешение уже было запрошено.
    fun open_camera() { //  открывает камеру с помощью CameraManager. Происходит асинхронное взаимодействие с камерой.
        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {
                override fun onOpened(p0: CameraDevice) {
                    cameraDevice = p0 // Сохраняется экземпляр открытой камеры

                    var surfaceTexture = textureView.surfaceTexture // Получается поверхность
                    var surface = Surface(surfaceTexture)

                    // Создаем запрос захвата изображения для предварительного просмотра
                    var captureRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequest.addTarget(surface)

                    // Создаем сессию захвата изображения
                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(p0: CameraCaptureSession) {
                                p0.setRepeatingRequest(captureRequest.build(), null, null)
                            }

                            override fun onConfigureFailed(p0: CameraCaptureSession) {
                            }
                        },
                        handler
                    )
                }

                override fun onDisconnected(p0: CameraDevice) { // Камера была отключена

                }

                override fun onError(
                    p0: CameraDevice,
                    p1: Int
                ) { // Произошла ошибка при открытии камеры

                }
            },
            handler
        )
    }

    fun get_permission() {
        // Проверяем, есть ли разрешение на использование камеры
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // Если разрешение отсутствует, запрашиваем его
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    //  отвечает за обработку запроса разрешений на использование камеры
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        // Проверяем, было ли дано разрешение на камеру
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {

            // Если разрешения нет - запрашиваем
            get_permission()
        }
    }
}