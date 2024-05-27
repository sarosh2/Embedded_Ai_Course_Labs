#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

#include "model.h"
#include "input.h"

const int numSamples = num_images;

int samplesRead = 0;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 50 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

#define NUM_DIGITS 10

void loadImageToInputTensor(TfLiteTensor* input, const int* image) {
  for (int i = 0; i < 28 * 28; ++i) {
    input->data.uint8[i] = static_cast<uint8_t>(image[i] / 255.0f * 255);
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // print out the number of input images
  Serial.print("Input Images = ");
  Serial.print(numSamples);
  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, nullptr);

  // Allocate memory for the model's input and output tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("Tensor Allocation Failed");
    return;
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  if (tflOutputTensor->dims->size != 2 || tflOutputTensor->dims->data[1] != 10) {
    Serial.println("Bad output tensor shape.");
    return;
  }

  Serial.println("Setup complete.");
  Serial.println();
}

void loop() {
  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {

    //read samples and add to input tensor
    loadImageToInputTensor(tflInputTensor, image_data[samplesRead]);
    samplesRead++;

    //run inferencing
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    tflOutputTensor = tflInterpreter->output(0);
    float scale = tflOutputTensor->params.scale;
    int zero_point = tflOutputTensor->params.zero_point;
    uint8_t* probabilities = tflOutputTensor->data.uint8;

    //print output
    Serial.print("Image No: ");
    Serial.println(samplesRead);
    float maxProb = -1.0;
    int maxDigit = -1;
    for (int i = 0; i < NUM_DIGITS; i++) {
      float probability = (probabilities[i] - zero_point) * scale;
      Serial.print(i);
      Serial.print(": ");
      Serial.println(probability, 6);
      if (probability > maxProb)
      {
        maxProb = probability;
        maxDigit = i;
      }
      
    }
    Serial.println();
    Serial.print("Predicted Digit: ");
    Serial.println(maxDigit);
    Serial.print("With predicted Probability: ");
    Serial.println(maxProb);
    Serial.println();
    delay(1000);
  }
  
  Serial.println("All Samples Read Successfully");
  delay(1000);

}