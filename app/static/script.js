document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    
    let fileInput = document.getElementById("fileInput").files[0];
    let disease = document.getElementById("disease").value;
    let predictionText = document.getElementById("predictionText");
    let uploadedImage = document.getElementById("uploadedImage");

    if (!fileInput) {
        alert("Please select an image before submitting.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput);
    formData.append("disease", disease);

    try {
        let response = await fetch("/predict", { method: "POST", body: formData });
        let data = await response.json();

        if (data.error) {
            predictionText.innerText = `Error: ${data.error}`;
            predictionText.style.color = "red";
            uploadedImage.style.display = "none";
        } else {
            predictionText.innerText = `Prediction: ${data.prediction}`;
            predictionText.style.color = "black";
            uploadedImage.src = data.image;
            uploadedImage.style.display = "block";
        }

        // Reset the file input
        document.getElementById("fileInput").value = "";
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing the request.");
    }
});

document.getElementById("generateReport").addEventListener("click", async function () {
    let doctorNotes = document.getElementById("doctorNotes").value.trim();
    let predictionTextElement = document.getElementById("predictionText").innerText;

    // Extract prediction text properly
    let prediction = predictionTextElement.includes("Prediction: ")
        ? predictionTextElement.replace("Prediction: ", "").trim()
        : "";

    let disease = document.getElementById("disease").value;

    if (!prediction || prediction.startsWith("Error")) {
        alert("Please make a valid prediction first.");
        return;
    }

    if (!doctorNotes) {
        alert("Please enter doctor's notes before generating a report.");
        return;
    }

    try {
        let response = await fetch("/generate_report", {
            method: "POST",
            body: JSON.stringify({ disease, prediction, doctor_notes: doctorNotes }),
            headers: { "Content-Type": "application/json" },
        });

        let data = await response.json();
        console.log("Server Response:", data);

        if (data.error) {
            alert("Error generating report: " + data.error);
        } else {
            let fullReport = data.report;

            // Remove text inside square brackets []
            fullReport = fullReport.replace(/\[.*?\]/g, "").trim();

            // Remove fields that are empty or have no meaningful data
            fullReport = fullReport
                .split("\n")
                .map(line => line.trim()) // Trim spaces
                .filter(line => line && !line.match(/:\s*$/)) // Remove empty fields
                .join("\n");

            // Store the cleaned-up report
            localStorage.setItem("generatedReport", JSON.stringify({ fullReport }));

            // Open the report in a new tab
            window.open("/view_report", "_blank");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while generating the report.");
    }
});
