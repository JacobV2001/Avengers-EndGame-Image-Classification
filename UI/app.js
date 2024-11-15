Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop your image here or click to upload",
        autoProcessQueue: false,
        acceptedFiles: "image/*", // restrict to images only
        init: function() {
            this.on("addedfile", function(file) {
                if (this.files[1] != null) {
                    this.removeFile(this.files[0]);
                }
            });
        }
    });

    // disable all when classifying
    dz.on("sending", function() {
        $("#submitBtn").attr("disabled", true).html('Classifying... <i class="fa fa-spinner fa-spin"></i>');
        $("#error").hide(); // hide previous error
        $("#resultHolder").hide(); // hide previous result
        $("#divClassTable").hide(); // hide previous table
        $("#loadingSpinner").show(); // show loading spinner
    });

    dz.on("complete", function(file) {
        let imageData = file.dataURL;

        var url = "http://127.0.0.1:5000/classify_image";

        $.post(url, {
            image_data: imageData
        }, function(data, status) {
            $("#loadingSpinner").hide();
            $("#submitBtn").attr("disabled", false).html('Classify');

            if (!data || data.length == 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            let players = ["captain_america", "doctor_strange", "gamora", "iron_man", "loki", "scarlet_witch", "spiderman", "thor"];
            let match = null;
            let bestScore = -1;

            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();


                let classDictionary = match.class_dictionary;
                let tableBody = $("#classTable tbody").empty();
                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    let rowHtml = `
                        <tr>
                            <td>${capitalizeFirstLetter(personName.replace('_', ' '))}</td>
                            <td>${(probabilityScore).toFixed(2)}%</td>
                        </tr>
                    `;
                    tableBody.append(rowHtml);
                }

                let characterHtml = `
                    <div class="card shadow-sm">
                        <img class="card-img-top rounded-circle" src="./images/${match.class}.jpg" alt="${match.class}">
                        <div class="card-body text-center">
                            <h5 class="card-title">${capitalizeFirstLetter(match.class.replace('_', ' '))}</h5>
                            <p class="text-muted">Match with ${Math.round(bestScore)}% confidence</p>
                        </div>
                    </div>
                `;
                $("#matchedCharacter").html(characterHtml);
            }
        }).fail(function() {
            $("#error").show();
            $("#loadingSpinner").hide();
            $("#submitBtn").attr("disabled", false).html('Classify');
        });
    });

    $("#submitBtn").on('click', function() {
        dz.processQueue();
    });
}

function capitalizeFirstLetter(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

$(document).ready(function() {
    console.log("App is ready!");

    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    $("#loadingSpinner").hide();

    init();
});
