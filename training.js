const path = require("path");
const fs = require("fs");
const fr = require("face-recognition");

const dataPath = path.resolve("./data/faces");
const classNames = ["sheldon", "lennard", "raj", "howard", "stuart"];

const allFiles = fs.readdirSync(dataPath);
const imagesByClass = classNames.map((c) =>
  allFiles
    .filter((f) => f.includes(c))
    .map((f) => path.join(dataPath, f))
    .map((fp) => fr.loadImage(fp))
);

const numTrainingFaces = 10;
const trainDataByClass = imagesByClass.map((imgs) =>
  imgs.slice(0, numTrainingFaces)
);
const testDataByClass = imagesByClass.map((imgs) =>
  imgs.slice(numTrainingFaces)
);

const image = fr.loadImage("image.png");
const detector = fr.FaceDetector();
const targetSize = 150;
const faceImages = detector.detectFaces(image, targetSize);
faceImages.forEach((img, i) => fr.saveImage(`face_${i}.png`, img));

const recognizer = fr.FaceRecognizer();

trainDataByClass.forEach((faces, label) => {
  const name = classNames[label];
  recognizer.addFaces(faces, name);
});

const modelState = recognizer.serialize();
fs.writeFileSync("model.json", JSON.stringify(modelState));
