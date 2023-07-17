## Process raw audio

```js
// Get the user's audio input.
// The browser will display the dialog if the use allows the application to access the microphone.
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({audio: true}).then(function (stream) {
        // From here, we have access to the user's audio input.
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioSource = audioContext.createMediaStreamSource(stream);

        // Load our processor by the file path.
        audioContext.audioWorklet.addModule("ourProcessor.js").then(
            function() {
                // The second parameter is the name we defined in the ourProcessor.js.
                // In that file, the name is defined by the registerProcessor() method.
                const ourProcessor = new AudioWorkletNode(audioContext, "ourProcessor");

                audioSource.connect(ourProcessor);
                ourProcessor.connect(audioContext.destination);
            },
            function (rejectedReason) {
                console.error(rejectedReason);
            }
        );
    });
}
```

Content of `ourProcessor.js`:
```js
/**
 * Class name can be anything as long as it extends {AudioWorkletProcessor}.
 */
class OurProcessor extends AudioWorkletProcessor {

    constructor() {
        super();
    }

    /**
     * We must create this method.
     * https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_AudioWorklet
     * @param inputList Number of active microphones.
     *                  The length is usually one because other microphones are deactivated.
     * @param outputList Number of speakers.
     * @param parameters Don't worry about this for now.
     * @return {boolean} We gotta return true unless we want to stop the audio processing permanently.
     */
    process(inputList, outputList, parameters) {
        // This limit is for preventing potential errors caused by accessing unavailable (non-existing) resources.
        const sourceLimit = Math.min(inputList.length, outputList.length);

        for (let inputIndex = 0; inputIndex < sourceLimit; inputIndex++) {
            const input = inputList[inputIndex];
            const output = outputList[inputIndex];

            // There are usually two channels (stereo). One for left ear, another for right.
            const channelLimit = Math.min(input.length, output.length);
            for (let channelIndex = 0; channelIndex < channelLimit; channelIndex++) {
                const inputChannel = input[channelIndex];
                const outputChannel = output[channelIndex];

                // There are 128 samples by the spec at the time of writing.
                // It might be changed in the future, though.
                // Note that the for-loop using index incrementation like above won't work
                // because inputChannel is not array. It's a JSON object where key is used as index like this:
                // {"0":-0.011914870701730251,"1":-0.01202797144651413, ... "127":-0.016677042469382286}
                for (let sampleIndex in inputChannel) {
                    // 32-bit floating point number
                    const inputSample = inputChannel[sampleIndex];

                    // play the sample by setting the output values.
                    outputChannel[sampleIndex] = inputSample;
                }
            }
        }
        return true;
    }

}

// We must call this built-in method.
// The first parameter can be any name.
// The name will be used when we add this as a module of an audioWorklet.
// The second parameter must be the class name above.
registerProcessor("ourProcessor", OurProcessor);
```

### Pass data from the audio processor to the main thread

Send data from the audio processor:
```js
class OurProcessor extends AudioWorkletProcessor {

    constructor() {
        super();
    }
    
    process(inputList, outputList, parameters) {
        // We can send any data via port.
        this.port.postMessage(inputList);
        
        return true;
    }

}

registerProcessor("ourProcessor", OurProcessor);
```

Receive the data:
```js
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({audio: true}).then(function (stream) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const audioSource = audioContext.createMediaStreamSource(stream);
        audioContext.audioWorklet.addModule("ourProcessor.js").then(
            function() {
                const ourProcessor = new AudioWorkletNode(audioContext, "ourProcessor");

                // Receive the data from the processor.
                ourProcessor.port.onmessage = function(event) {
                    // Print the data.
                    console.log(event.data);
                }

                audioSource.connect(ourProcessor);
                ourProcessor.connect(audioContext.destination);
            },
            function (rejectedReason) {
                console.error(rejectedReason);
            }
        );
    });
}
```
