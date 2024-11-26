const express = require("express");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");
const tf = require("@tensorflow/tfjs-node");
const { loadModel: loadModelFromInference, predict } = require('./inference');
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 8080;

// Multer setup for file uploads with size limit of 1MB
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    // Set the destination folder where the file will be saved
    const uploadsDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir); // Create the folder if it does not exist
    }
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    // Keep the original file name
    cb(null, file.originalname);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 1000000 }, // Max 1MB
});

// Load model from Google Cloud Storage or local
let model;
async function loadModel() {
  try {
    model = await loadModelFromInference(); // Assuming loadModel is imported from inference.js
    console.log("Model loaded successfully!");
  } catch (err) {
    console.error("Error loading model:", err.message);
  }
}
loadModel();

// POST /predict endpoint
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    // Validate file
    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "No image file provided",
      });
    }

    console.log('Uploaded file:', req.file); // Log file information

    // Load and preprocess image
    const imageBuffer = fs.readFileSync(req.file.path); // Read the image buffer
    const imageTensor = tf.node.decodeImage(imageBuffer).resizeBilinear([224, 224]).expandDims(0);

    // Predict
    const predictions = model.predict(imageTensor);
    const result = predictions.dataSync()[0]; // Assume binary classification (0 or 1)

    // Build response
    const response = {
      status: "success",
      message: "Model is predicted successfully",
      data: {
        id: uuidv4(),
        result: result > 0.5 ? "Cancer" : "Non-cancer",
        suggestion: result > 0.5 ? "Segera periksa ke dokter!" : "Penyakit kanker tidak terdeteksi.",
        createdAt: new Date().toISOString(),
      },
    };
    res.json(response);
  } catch (err) {
    console.error(err);
    res.status(400).json({
      status: "fail",
      message: "Terjadi kesalahan dalam melakukan prediksi",
    });
  }
});

// Error handling for file size
app.use((err, req, res, next) => {
  if (err.code === "LIMIT_FILE_SIZE") {
    res.status(413).json({
      status: "fail",
      message: "File terlalu besar. Maksimal ukuran file adalah 1MB.",
    });
  } else {
    next(err);
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
