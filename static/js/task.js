/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */


    // Initalize psiturk object
    var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

    function split_joint_counterbalance(joint_counterbalance) {
        var train_test_set = 0;

        if (joint_counterbalance <= 5) {
            train_test_set = 0;
            counterbalance = joint_counterbalance;
        }
        else if (joint_counterbalance >= 12) {
            train_test_set = 2;
            counterbalance = joint_counterbalance - 12;
        }
        else {
            train_test_set = 1;
            counterbalance = joint_counterbalance - 6;
        }

        return [counterbalance, train_test_set]
    }

    // condition will determine the condition, counterbalance will determine domain order, number of data points will determine which train/test set you get

    // a) manually assign from remaining conditions
    // var remaining_joint_conditions = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 3, 0], [0, 3, 1], [0, 3, 2], [0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 5, 0], [0, 5, 1], [0, 5, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [1, 3, 0], [1, 3, 1], [1, 3, 2], [1, 4, 0], [1, 4, 1], [1, 4, 2], [1, 5, 0], [1, 5, 1], [1, 5, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2], [2, 3, 0], [2, 3, 1], [2, 3, 2], [2, 4, 0], [2, 4, 1], [2, 4, 2], [2, 5, 0], [2, 5, 1], [2, 5, 2], [3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 1, 0], [3, 1, 1], [3, 1, 2], [3, 2, 0], [3, 2, 1], [3, 2, 2], [3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 4, 0], [3, 4, 1], [3, 4, 2], [3, 5, 0], [3, 5, 1], [3, 5, 2], [4, 0, 0], [4, 0, 1], [4, 0, 2], [4, 1, 0], [4, 1, 1], [4, 1, 2], [4, 2, 0], [4, 2, 1], [4, 2, 2], [4, 3, 0], [4, 3, 1], [4, 3, 2], [4, 4, 0], [4, 4, 1], [4, 4, 2], [4, 5, 0], [4, 5, 1], [4, 5, 2], [5, 0, 0], [5, 0, 1], [5, 0, 2], [5, 1, 0], [5, 1, 1], [5, 1, 2], [5, 2, 0], [5, 2, 1], [5, 2, 2], [5, 3, 0], [5, 3, 1], [5, 3, 2], [5, 4, 0], [5, 4, 1], [5, 4, 2], [5, 5, 0], [5, 5, 1], [5, 5, 2], [6, 0, 0], [6, 0, 1], [6, 0, 2], [6, 1, 0], [6, 1, 1], [6, 1, 2], [6, 2, 0], [6, 2, 1], [6, 2, 2], [6, 3, 0], [6, 3, 1], [6, 3, 2], [6, 4, 0], [6, 4, 1], [6, 4, 2], [6, 5, 0], [6, 5, 1], [6, 5, 2], [7, 0, 0], [7, 0, 1], [7, 0, 2], [7, 1, 0], [7, 1, 1], [7, 1, 2], [7, 2, 0], [7, 2, 1], [7, 2, 2], [7, 3, 0], [7, 3, 1], [7, 3, 2], [7, 4, 0], [7, 4, 1], [7, 4, 2], [7, 5, 0], [7, 5, 1], [7, 5, 2], [8, 0, 0], [8, 0, 1], [8, 0, 2], [8, 1, 0], [8, 1, 1], [8, 1, 2], [8, 2, 0], [8, 2, 1], [8, 2, 2], [8, 3, 0], [8, 3, 1], [8, 3, 2], [8, 4, 0], [8, 4, 1], [8, 4, 2], [8, 5, 0], [8, 5, 1], [8, 5, 2]];
    // var joint_condition = remaining_joint_conditions[Math.floor(Math.random() * remaining_joint_conditions.length)];
    // condition = joint_condition[0];
    // counterbalance = joint_condition[1];
    // var train_test_set = joint_condition[2];

    // b) rely on PsiTurk's counterbalancing
    var split_counterbalance = split_joint_counterbalance(counterbalance);
    counterbalance = split_counterbalance[0];
    var train_test_set = split_counterbalance[1];

    // manual testing
    // condition = 0;
    // counterbalance = 0;
    // var train_test_set = 0;

    console.log(condition);
    console.log(counterbalance);
    console.log(train_test_set);

    // Headers
    var training_video_header = "<h1>Demonstration of best strategy</h1> <hr/>" + "<h4>Feel free to watch the video as many times as you want (but at least once)!</h4> ";
    var training_video_header_final = "<h1>Demonstration of best strategy</h1> <hr/>" + "<h4>Feel free to watch the video as many times as you want (but at least once)!</h4>" + "<br><h4>Note: This the <u>last chance</u> to review any previous demonstrations (by pressing on the 'Previous' button below) if you wish!</h4>" +
        "<h4>(You will be asked about the informativeness and difficulty of the demonstrations on the following page.)</h4>"

    var testing_simulation_header = "<h1>Participant gameplay</h1> <hr/>" + "<h3>Now it's your turn to play six games! Please demonstrate how you would complete this game's task while minimizing Chip's energy loss.</h3> " +
        "You <b>may reset</b> at any time <b>before the game ends</b> if think you've made a mistake.<br> <h3>Please answer the question below on the confidence of your gameplay <b>after</b> providing your demonstration.</h3>" +
        "For fun: We will let you know how well you did at the end of all six games."

    var sandbox_2_header = "<h1>Free play</h1> <hr/> " +
        "<h4>A subset of the following keys will be available to control Chip in each game:</h4>" +
        "<table class=\"center\"><tr><th>Key</th><th>Action</th></tr><tr><td>up/down/left/right arrow keys</td><td>corresponding movement</td></tr><tr><td>p</td><td>pick up</td></tr><tr><td>d</td><td>drop</td></tr><tr><td>r</td><td>reset simulation</td></tr></table><br>" +
        "<h4>If you accidentally take a wrong action, you may reset the simulation and start over.</h4> <br>" +
        "<h3>Feel free to play around in the game below and get used to the controls. </h3> <h4>You can click the continue button whenever you feel ready to move on.</h4>";


    var sandbox_3_header = "<h1>Practice game</h1> <hr/> " +
        "<h3>As previously mentioned, the task in this practice game is the following: </h3> <br>" +
        "<table class=\"center\"><tr><th>Task</th><th>Sample sequence</th></tr><tr><td>Dropping off the green pentagon at the purple star</td><td><img src = 'static/img/sandbox_dropoff1.png' width=\"75\" height=auto /><img src = 'static/img/arrow.png' width=\"30\" height=auto /><img src = 'static/img/sandbox_dropoff2.png' width=\"75\" height=auto /></td></tr></table> <br>" +
        "<h3>Each game will consist of <b>actions that change your energy level</b> differently. In this game, the following actions affect your energy:</h3> <br>" +
        "<table class=\"center\"><tr><th>Action</th><th>Sample sequence</th><th>Energy change</th></tr>" +
        "<tr><td>Moving through the orange diamond</td><td><img src = 'static/img/sandbox_diamond1.png' width=\"225\" height=auto /><img src = 'static/img/arrow.png' width=\"30\" height=auto /><img src = 'static/img/sandbox_diamond2.png' width=\"225\" height=auto /> <img src='static/img/arrow.png' width=\"30\" height=auto /><img src ='static/img/sandbox_diamond3.png' width=\"225\" height=auto/> <td>+10%</td></tr>" +
        "<tr><td>Any action that you take (e.g. moving right)</td><td><img src = 'static/img/right1.png' width=\"150\" height=auto /><img src = 'static/img/arrow.png' width=\"30\" height=auto /><img src = 'static/img/right2.png' width=\"150\" height=auto /><td>-5%</td></tr></table> <br>" +
        "<h3><b>Pick up the green pentagon</b> and <b>drop it off at the purple star</b> with the <b>maximum possible energy remaining</b>. </h3> " +
        "<h4>You should end with 40% energy left (you won't be able to move if energy falls to 0%). <u>You will have 3 chances to get it right to continue on with the study!</u></h4>" +
        "<h4>Note: Since this is practice, we have revealed each actions's effect on Chip's energy and also provide a running counter of Chip's current energy level below.</h4> <br>";

    var post_main_game_header = "<h3>You've completed all of the gameplays for this game! </h3><h3>If you were curious, you got TEMP_SCORE out of 6 games correct.</h3><br><h3>Let's head over to the next one!</h3><br>"
    var post_main_game_header_final = "<h3>You've completed all of the gameplays for this final game! If you were curious, you got TEMP_SCORE out of 6 games correct.</h3><br>"

    // -- Surveys
    var interim_header = "You have viewed all of the demonstrations for this game! Please complete the survey below.";
    var post_study_header = "<h2>Congratulations on getting through all of the main games!</h2><br>Please answer the following questions, and then click ‘Continue’ for a closing debrief on the study and instructions on compensation.";

    var training_informativeness_prompt = "How informative were these demonstrations in understanding the best strategy in this game?";
    var training_mental_effort_prompt = "How much mental effort was required to understand the best strategy in this game?";
    var testing_difficulty_prompt = "How confident are you that you minimized Chip's energy loss while completing the task?";

    // INSTRUCTIONS
    var instructions_intro = {
        type:'external-html',
        url: "instructions/intro.html",
        question_type: "study_introduction",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var instructions_overview = {
        type:'external-html',
        url: "instructions/overview.html",
        question_type: "overview",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var sandbox_introduction = {
        type:'external-html',
        url: "instructions/sandbox_introduction.html",
        question_type: "sandbox_introduction",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var augmented_taxi2_introduction = {
        type:'external-html',
        url: "instructions/augmented_taxi2_introduction.html",
        question_type: "augmented_taxi2_introduction",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var colored_tiles_introduction = {
        type:'external-html',
        url: "instructions/colored_tiles_introduction.html",
        question_type: "colored_tiles_introduction",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var skateboard2_introduction = {
        type:'external-html',
        url: "instructions/skateboard2_introduction.html",
        question_type: "skateboard2_introduction",
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };

    var post_practice_button = {
        type: "html-button-response",
        stimulus: "<h3>Good job on completing the practice game! Let's now head over to the three main games and <b>begin the real study</b>.</h3><br>" +
            "<h3>In these games, you will <b>not</b> be told how each action changes Chip's energy level.</h3><br>" +
            "For example, note the '???' in the Energy Change column below. <table class=\"center\"><tr><th>Action</th><th>Sample sequence</th><th>Energy change</th></tr><tr><td>Any action that you take (e.g. moving right)</td><td><img src = 'static/img/right1.png' width=\"150\" height=auto /><img src = 'static/img/arrow.png' width=\"30\" height=auto /><img src = 'static/img/right2.png' width=\"150\" height=auto /><td>???</td></tr></table> <br>" +
            "<h3>Instead, you will have to <u>figure that out</u> and subsequently the best strategy for completing the task while minimizing Chip's energy loss <u>by observing Chip's demonstrations!</u></h3><br>",
        question_type: "post_practice_stage",
        choices: ["Continue"],
        domain: 'sandbox',
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };


    // Miscellaneous pages
    var debrief = {
        type:'external-html',
        url: "debrief.html",
        question_type: 'debrief',
        execute_script: true,
        force_refresh: true,
        cont_btn: "next",
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };



    //LIKERT SCALES
    var training_informativeness_scale = [
        "Not Informative",
        "Slightly Informative",
        "Moderately Informative",
        "Very Informative",
        "Extremely Informative"
    ];
    var training_mental_effort_scale = [
        "No Effort",
        "Slight Effort",
        "Moderate Effort",
        "Significant Effort",
        "Extreme Effort"
    ];
    var testing_difficulty_scale = [
        "Not Confident",
        "Slightly Confident",
        "Somewhat Confident",
        "Very Confident",
        "Extremely Confident",
    ];

    var sandbox_1_parameters = {
        'agent': {'x': 4, 'y': 1, 'has_passenger': 0},
        'walls': [{'x': 1, 'y': 3}, {'x': 2, 'y': 3}, {'x': 3, 'y': 3}],
        'passengers': [{'x': 1, 'y': 2, 'dest_x': 1, 'dest_y': 4, 'in_taxi': 0}],
        'hotswap_station': [{'x': 2, 'y': 1}],
        'width': 4,
        'height': 4,
    };

    var sandbox_2_parameters = {
        'agent': {'x': 4, 'y': 3, 'has_passenger': 0},
        'walls': [{'x': 2, 'y': 3}, {'x': 2, 'y': 2}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2}],
        'passengers': [{'x': 4, 'y': 1, 'dest_x': 1, 'dest_y': 4, 'in_taxi': 0}],
        'hotswap_station': [{'x': 1, 'y': 2}],
        'width': 4,
        'height': 4,
    };


var post_study_survey = {
    type: 'survey-text',
    preamble: post_study_header,
    questions: [
        {
            prompt: "Do you have any comments or feedback on the study?",
            qtype: "text",
            placeholder: "Your answer here...",
            rows: 5
        },
        {
            prompt: "What is your age?",
            qtype: "number",
            placeholder: "Your answer here...",
            required: true,
            rows: 1
        },
        {
            prompt: "Which of the following most accurately describes you?",
            qtype: "multi",
            options: ["Male", "Female", "Non-binary", "Prefer not to disclose"],
            required: true,
        },
    ],
    force_refresh: true,
    execute_script: true,
    question_type: 'post_study_survey',
    condition: condition,
    counterbalance: counterbalance,
    train_test_set: train_test_set
};

function testing_simulation(source, mdp_parameters, domain, preamble, continue_condition = null) {
    if (domain == 'colored_tiles'){
        key_table = "<br><br><br><table class=\"center\"><tr><th>Key</th><th>Action</th></tr><tr><td>up/down/left/right arrow keys</td><td>corresponding movement</td></tr><tr><td>r</td><td>reset simulation</td></tr></table><br>";
    } else {
        key_table = "<br><br><br><table class=\"center\"><tr><th>Key</th><th>Action</th></tr><tr><td>up/down/left/right arrow keys</td><td>corresponding movement</td></tr><tr><td>p</td><td>pick up</td></tr><tr><td>d</td><td>drop</td></tr><tr><td>r</td><td>reset simulation</td></tr></table><br>";
    }


    if (domain == 'sandbox') {
        stimulus = '<iframe id = "ifrm" style="border:none;" src="' + source + '" height="550" width="950" title="Iframe Example"></iframe>    ';
        questions = []
    }
    else if (domain == 'augmented_taxi2') {
        stimulus = '<iframe id = "ifrm" style="border:none;" src="' + source + '" height="550" width="750" title="Iframe Example"></iframe>    '
        questions = [{prompt: testing_difficulty_prompt, labels: testing_difficulty_scale, required: true }]
    } else if (domain == 'colored_tiles') {
        stimulus = '<iframe id = "ifrm" style="border:none;" src="' + source + '" height="700" width="700" title="Iframe Example"></iframe>    '
        questions = [{prompt: testing_difficulty_prompt, labels: testing_difficulty_scale, required: true }]
    }
    else {
        stimulus = '<iframe id = "ifrm" style="border:none;" src="' + source + '" height="600" width="800" title="Iframe Example"></iframe>    '
        questions = [{prompt: testing_difficulty_prompt, labels: testing_difficulty_scale, required: true }]
    }

    if (domain == 'sandbox' && continue_condition == 'free_play') {
        legend = null;
    } else {
        legend = key_table;
    }

    var ret = {
        // Webpage information
        type: 'sim-likert',
        preamble: preamble,
        legend: legend,
        iframe_id: "ifrm",
        stimulus: stimulus,
        questions: questions,

        // Testing information
        mdp_parameters: mdp_parameters,

        // Study information
        domain: domain,
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set,
        question_type: 'testing_simulation',

        // for sandbox domain
        continue_condition: continue_condition
    };
    return ret;
}

function training_video(primary_videos, domain, preamble = training_video_header) {

    var myhtml = '<div id="jspsych-html-preamble">'+training_video_header+'</div>';
    var myhtml_final = '<div id="jspsych-html-preamble">'+training_video_header_final +'</div>';

    pages = [];
    if (domain == 'augmented_taxi2') {
        for (var i = 0; i < primary_videos.length; i++) {
            if (i == primary_videos.length - 1) pages.push(myhtml_final + '<video controls width="716" height="538" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
            else pages.push(myhtml + '<video controls width="716" height="538" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
        }
    } else if (domain == 'colored_tiles') {
        for (var i = 0; i < primary_videos.length; i++) {
            if (i == primary_videos.length - 1) pages.push(myhtml_final + '<video controls width="700" height="700" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
            else pages.push(myhtml + '<video controls width="700" height="700" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
        }
    }
    else {
        for (var i = 0; i < primary_videos.length; i++) {
            if (i == primary_videos.length - 1) pages.push(myhtml_final + '<video controls width="1072" height="714" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
            else pages.push(myhtml + '<video controls width="1072" height="714" id="stimulus" src="' + primary_videos[i] + '"type="video/mp4" > </video>');
        }
    }

    var ret = {
        // Webpage information
        type: 'sim-instructions',
        pages: pages,
        show_clickable_nav: true,

        // Testing information
        mdp_parameters: {},

        // Study information
        domain: domain,
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set,
        question_type: 'training_video',
    };

    return ret;
}

function post_main_game(stimulus, domain) {
    var post_main_game_button = {
        type: "html-button-response",
        stimulus: stimulus,
        question_type: "post_main_game",
        choices: ["Continue"],
        domain: domain,
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };
    return post_main_game_button
}


function training_survey(domain) {
    var survey = {
        type: 'survey-likert',
        preamble: interim_header,
        questions: [
            {prompt: training_informativeness_prompt, labels: training_informativeness_scale, required: true },
            {prompt: training_mental_effort_prompt, labels: training_mental_effort_scale, required: true }
        ],
        freeform:
            [{
                prompt: "Feel free to explain any of your selections above if you wish:",
                qtype: "text",
                placeholder: "Your answer here...",
                rows: 5,
                columns: 40
          }],
        question_type: 'interim_survey',
        randomize_question_order: false,
        domain: domain,
        condition: condition,
        counterbalance: counterbalance,
        train_test_set: train_test_set
    };
    return survey
}

    // https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
    function shuffle(array) {
      var currentIndex = array.length, temporaryValue, randomIndex;

      // While there remain elements to shuffle...
      while (0 !== currentIndex) {

        // Pick a remaining element...
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex -= 1;

        // And swap it with the current element.
        temporaryValue = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = temporaryValue;
      }

      return array;
    }

    var augmented_taxi2_testing = [], colored_tiles_testing = [], skateboard2_testing = []

    /* 3 sets of testing demos were randomly selected to ensure that no one uncharacteristic selection
     could heavily affect the results.  */

    var training_mapping;
    if (train_test_set == 0) {
        training_mapping = 'a';
    } else if (train_test_set == 1) {
        training_mapping = 'b';
    } else {
        training_mapping = 'c';
    }

    // 0: baseline, 1: counterfactual only, 2: feature only, 3: proposed
    if (condition == 0) {
        var folder = 'baseline';
    } else if (condition == 1) {
        var folder = 'counterfactual_only';
    } else if (condition == 2) {
        var folder = 'feature_only';
    } else if (condition == 3) {
        var folder = 'proposed';
    } else {
        console.error('Unexpected condition encountered.');
    }

    // load the names of the training videos based on the chosen condition
    var augmented_taxi2_videos = [], colored_tiles_videos = [], skateboard2_videos = []
    for (var i = 1; i < 6; i++) {
        augmented_taxi2_videos.push('/static/vid/' + folder + '/taxi_' + i.toString() + '.mp4')
    }
    for (var i = 1; i < 6; i++) {
        colored_tiles_videos.push('/static/vid/' + folder + '/colored_tiles_' + i.toString() + '.mp4')
    }
    for (var i = 1; i < 8; i++) {
        skateboard2_videos.push('/static/vid/' + folder + '/skateboard_' + i.toString() + '.mp4')
    }

    var testing_mdp_parameters = {"augmented_taxi2": {"low": [[{"agent": {"x": 1, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 0, 0, 0, 0], "opt_actions": ["right", "right", "right", "pickup", "left", "left", "left", "dropoff"], "opt_traj_length": 8, "opt_traj_reward": -1.695996608010176, "test_difficulty": "low", "tag": 0, "all_opt_actions": [["right", "right", "right", "pickup", "left", "left", "left", "dropoff"]]}, {"agent": {"x": 4, "y": 3, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 3}, {"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [1, 0, 1, 1, 0, 1], "opt_actions": ["down", "down", "pickup", "left", "left", "left", "dropoff"], "opt_traj_length": 7, "opt_traj_reward": -1.377997244008268, "test_difficulty": "low", "tag": 1, "all_opt_actions": [["down", "down", "pickup", "left", "left", "left", "dropoff"]]}], [{"agent": {"x": 2, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 2, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 0, 0, 1, 1], "opt_actions": ["up", "pickup", "down", "left", "dropoff"], "opt_traj_length": 5, "opt_traj_reward": -1.05999788000636, "test_difficulty": "low", "tag": 2, "all_opt_actions": [["up", "pickup", "down", "left", "dropoff"]]}, {"agent": {"x": 4, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 0, 0, 1, 1], "opt_actions": ["up", "up", "pickup", "down", "down", "left", "left", "left", "dropoff"], "opt_traj_length": 9, "opt_traj_reward": -2.5439949120152643, "test_difficulty": "low", "tag": 3, "all_opt_actions": [["up", "up", "pickup", "down", "down", "left", "left", "left", "dropoff"], ["up", "up", "pickup", "down", "left", "down", "left", "left", "dropoff"], ["up", "up", "pickup", "down", "left", "left", "down", "left", "dropoff"], ["up", "up", "pickup", "left", "down", "down", "left", "left", "dropoff"], ["up", "up", "pickup", "left", "down", "left", "down", "left", "dropoff"], ["up", "up", "pickup", "left", "left", "down", "down", "left", "dropoff"]]}], [{"agent": {"x": 2, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 0, 0, 0, 1], "opt_actions": ["right", "right", "pickup", "left", "left", "left", "dropoff"], "opt_traj_length": 7, "opt_traj_reward": -1.483997032008904, "test_difficulty": "low", "tag": 4, "all_opt_actions": [["right", "right", "pickup", "left", "left", "left", "dropoff"]]}, {"agent": {"x": 4, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 3, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 0], "opt_actions": ["down", "left", "pickup", "left", "left", "dropoff"], "opt_traj_length": 6, "opt_traj_reward": -1.907996184011448, "test_difficulty": "low", "tag": 5, "all_opt_actions": [["down", "left", "pickup", "left", "left", "dropoff"]]}]], "medium": [[{"agent": {"x": 4, "y": 3, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}, {"x": 3, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 1, 0, 0, 1], "opt_actions": ["left", "left", "pickup", "down", "down", "left", "dropoff"], "opt_traj_length": 7, "opt_traj_reward": -1.377997244008268, "test_difficulty": "medium", "tag": 0, "all_opt_actions": [["left", "left", "pickup", "down", "down", "left", "dropoff"]]}, {"agent": {"x": 1, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 3, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 4, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 0, 1, 1, 1], "opt_actions": ["right", "right", "pickup", "left", "left", "dropoff"], "opt_traj_length": 6, "opt_traj_reward": -1.907996184011448, "test_difficulty": "medium", "tag": 1, "all_opt_actions": [["right", "right", "pickup", "left", "left", "dropoff"]]}], [{"agent": {"x": 4, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}, {"x": 3, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 1, 0, 0, 1], "opt_actions": ["up", "left", "left", "pickup", "down", "down", "left", "dropoff"], "opt_traj_length": 8, "opt_traj_reward": -2.3319953360139922, "test_difficulty": "medium", "tag": 2, "all_opt_actions": [["up", "left", "left", "pickup", "down", "down", "left", "dropoff"]]}, {"agent": {"x": 2, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 0, 1, 1], "opt_actions": ["up", "up", "pickup", "down", "down", "left", "dropoff"], "opt_traj_length": 7, "opt_traj_reward": -1.483997032008904, "test_difficulty": "medium", "tag": 3, "all_opt_actions": [["up", "up", "pickup", "down", "down", "left", "dropoff"]]}], [{"agent": {"x": 3, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}, {"x": 3, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 1, 0, 0, 1], "opt_actions": ["down", "left", "pickup", "left", "dropoff"], "opt_traj_length": 5, "opt_traj_reward": -1.695996608010176, "test_difficulty": "medium", "tag": 4, "all_opt_actions": [["down", "left", "pickup", "left", "dropoff"]]}, {"agent": {"x": 2, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 3}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [1, 0, 0, 0, 1, 1], "opt_actions": ["up", "up", "pickup", "down", "down", "left", "dropoff"], "opt_traj_length": 7, "opt_traj_reward": -1.483997032008904, "test_difficulty": "medium", "tag": 5, "all_opt_actions": [["up", "up", "pickup", "down", "down", "left", "dropoff"]]}]], "high": [[{"agent": {"x": 4, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 3, "y": 2, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 1], "opt_actions": ["up", "left", "down", "pickup", "down", "left", "left", "dropoff"], "opt_traj_length": 8, "opt_traj_reward": -2.225995548013356, "test_difficulty": "high", "tag": 0, "all_opt_actions": [["up", "left", "down", "pickup", "down", "left", "left", "dropoff"], ["up", "left", "down", "pickup", "left", "down", "left", "dropoff"]]}, {"agent": {"x": 3, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 1, 1, 1, 1, 1], "opt_actions": ["up", "right", "down", "down", "pickup", "left", "left", "left", "dropoff"], "opt_traj_length": 9, "opt_traj_reward": -3.073993852018444, "test_difficulty": "high", "tag": 1, "all_opt_actions": [["up", "right", "down", "down", "pickup", "left", "left", "left", "dropoff"]]}], [{"agent": {"x": 3, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 3, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 0, 1, 0], "opt_actions": ["left", "up", "up", "right", "pickup", "left", "down", "down", "left", "dropoff"], "opt_traj_length": 10, "opt_traj_reward": -2.7559944880165363, "test_difficulty": "high", "tag": 2, "all_opt_actions": [["left", "up", "up", "right", "pickup", "left", "down", "down", "left", "dropoff"], ["right", "up", "up", "left", "pickup", "left", "down", "down", "left", "dropoff"]]}, {"agent": {"x": 3, "y": 2, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 2, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 1], "opt_actions": ["up", "right", "down", "pickup", "down", "left", "left", "left", "dropoff"], "opt_traj_length": 9, "opt_traj_reward": -2.4379951240146283, "test_difficulty": "high", "tag": 3, "all_opt_actions": [["up", "right", "down", "pickup", "down", "left", "left", "left", "dropoff"], ["right", "pickup", "up", "left", "left", "down", "down", "left", "dropoff"]]}], [{"agent": {"x": 3, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 3, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 1], "opt_actions": ["left", "up", "up", "right", "pickup", "left", "down", "down", "left", "dropoff"], "opt_traj_length": 10, "opt_traj_reward": -2.11999576001272, "test_difficulty": "high", "tag": 4, "all_opt_actions": [["left", "up", "up", "right", "pickup", "left", "down", "down", "left", "dropoff"]]}, {"agent": {"x": 4, "y": 1, "has_passenger": 0}, "walls": [{"x": 1, "y": 3}, {"x": 1, "y": 2}], "passengers": [{"x": 4, "y": 2, "dest_x": 1, "dest_y": 1, "in_taxi": 0}], "tolls": [{"x": 3, "y": 3}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "available_tolls": [{"x": 3, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 3, "y": 1}], "traffic": [], "fuel_station": [], "hotswap_station": [{"x": 4, "y": 3}], "available_hotswap_stations": [{"x": 4, "y": 3}], "width": 4, "height": 3, "gamma": 1, "env_code": [1, 0, 1, 1, 1, 1], "opt_actions": ["up", "pickup", "up", "left", "left", "down", "down", "left", "dropoff"], "opt_traj_length": 9, "opt_traj_reward": -2.4379951240146283, "test_difficulty": "high", "tag": 5, "all_opt_actions": [["up", "pickup", "up", "left", "left", "down", "down", "left", "dropoff"]]}]]}, "colored_tiles": {"low": [[{"agent": {"x": 4, "y": 2}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 0, 1, 1, 1, 0, 1, 1], "opt_actions": ["right", "down"], "opt_traj_length": 2, "opt_traj_reward": -0.2376703644089226, "test_difficulty": "low", "tag": 0, "all_opt_actions": [["right", "down"]]}, {"agent": {"x": 2, "y": 3}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 1, 1, 1, 1, 1, 1, 0], "opt_actions": ["right", "right", "right", "down", "down"], "opt_traj_length": 5, "opt_traj_reward": -0.5941759110223065, "test_difficulty": "low", "tag": 1, "all_opt_actions": [["right", "right", "right", "down", "down"]]}], [{"agent": {"x": 2, "y": 2}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 1, 0, 0, 0, 1, 1, 1], "opt_actions": ["down", "right", "right", "right"], "opt_traj_length": 4, "opt_traj_reward": -0.4753407288178452, "test_difficulty": "low", "tag": 2, "all_opt_actions": [["down", "right", "right", "right"]]}, {"agent": {"x": 3, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 0, 0, 1, 1, 0, 1, 1], "opt_actions": ["right", "down", "down", "down", "down", "right"], "opt_traj_length": 6, "opt_traj_reward": -0.7130110932267678, "test_difficulty": "low", "tag": 3, "all_opt_actions": [["right", "down", "down", "down", "down", "right"], ["right", "down", "down", "down", "right", "down"]]}], [{"agent": {"x": 5, "y": 2}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 5, "y": 4}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 0, 0, 1, 0], "opt_actions": ["down"], "opt_traj_length": 1, "opt_traj_reward": -0.1188351822044613, "test_difficulty": "low", "tag": 4, "all_opt_actions": [["down"]]}, {"agent": {"x": 5, "y": 3}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 0, 1, 1, 0, 1, 1, 1], "opt_actions": ["down", "down"], "opt_traj_length": 2, "opt_traj_reward": -0.8615550709823445, "test_difficulty": "low", "tag": 5, "all_opt_actions": [["down", "down"]]}]], "medium": [[{"agent": {"x": 2, "y": 4}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 1, 1, 0, 1, 1, 0, 1], "opt_actions": ["down", "right", "right", "right", "down", "down"], "opt_traj_length": 6, "opt_traj_reward": -1.3368957998001898, "test_difficulty": "medium", "tag": 0, "all_opt_actions": [["down", "right", "right", "right", "down", "down"]]}, {"agent": {"x": 4, "y": 4}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 1, 1, 0, 0, 1, 1, 0, 1], "opt_actions": ["down", "right", "down", "down"], "opt_traj_length": 4, "opt_traj_reward": -1.723110141964689, "test_difficulty": "medium", "tag": 1, "all_opt_actions": [["down", "right", "down", "down"], ["right", "down", "down", "down"]]}], [{"agent": {"x": 1, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 1, 1, 1, 1, 1, 1, 1, 1], "opt_actions": ["down", "down", "right", "right", "right", "right", "down", "down"], "opt_traj_length": 8, "opt_traj_reward": -1.5745661642091124, "test_difficulty": "medium", "tag": 2, "all_opt_actions": [["down", "down", "right", "right", "right", "right", "down", "down"]]}, {"agent": {"x": 2, "y": 4}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 0, 1, 1, 1, 1, 1, 0, 1], "opt_actions": ["down", "right", "right", "right", "down", "down"], "opt_traj_length": 6, "opt_traj_reward": -1.9607805063736117, "test_difficulty": "medium", "tag": 3, "all_opt_actions": [["down", "right", "right", "right", "down", "down"]]}], [{"agent": {"x": 2, "y": 4}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 1, 1, 1, 0, 0, 0, 0, 1], "opt_actions": ["down", "right", "right", "right", "down", "down"], "opt_traj_length": 6, "opt_traj_reward": -1.3368957998001898, "test_difficulty": "medium", "tag": 4, "all_opt_actions": [["down", "right", "right", "right", "down", "down"], ["right", "down", "right", "right", "down", "down"], ["right", "right", "down", "right", "down", "down"], ["right", "right", "right", "down", "down", "down"]]}, {"agent": {"x": 4, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 0, 1, 1, 0, 0, 0, 0, 1], "opt_actions": ["down", "down", "right", "down", "down"], "opt_traj_length": 5, "opt_traj_reward": -1.2180606175957283, "test_difficulty": "medium", "tag": 5, "all_opt_actions": [["down", "down", "right", "down", "down"], ["down", "right", "down", "down", "down"], ["right", "down", "down", "down", "down"]]}]], "high": [[{"agent": {"x": 4, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 1, 1, 0, 0, 0, 1, 1, 1], "opt_actions": ["left", "down", "down", "left", "left", "down", "down", "right", "right", "right", "right"], "opt_traj_length": 11, "opt_traj_reward": -1.3071870042490743, "test_difficulty": "high", "tag": 0, "all_opt_actions": [["left", "down", "down", "left", "left", "down", "down", "right", "right", "right", "right"], ["left", "down", "left", "down", "left", "down", "down", "right", "right", "right", "right"], ["left", "down", "left", "left", "down", "down", "down", "right", "right", "right", "right"], ["left", "left", "down", "down", "left", "down", "down", "right", "right", "right", "right"], ["left", "left", "down", "left", "down", "down", "down", "right", "right", "right", "right"], ["left", "left", "left", "down", "down", "down", "down", "right", "right", "right", "right"]]}, {"agent": {"x": 4, "y": 3}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 3, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [0, 0, 1, 0, 0, 1, 0, 1, 1], "opt_actions": ["left", "down", "down", "right", "right"], "opt_traj_length": 5, "opt_traj_reward": -0.5941759110223065, "test_difficulty": "high", "tag": 1, "all_opt_actions": [["left", "down", "down", "right", "right"]]}], [{"agent": {"x": 4, "y": 3}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 0, 1, 0, 0, 1, 1, 1, 1], "opt_actions": ["left", "down", "down", "right", "right"], "opt_traj_length": 5, "opt_traj_reward": -0.5941759110223065, "test_difficulty": "high", "tag": 2, "all_opt_actions": [["left", "down", "down", "right", "right"]]}, {"agent": {"x": 4, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 1, 0, 1, 0, 1, 1, 1, 1], "opt_actions": ["left", "left", "down", "down", "right", "right", "down", "right", "down"], "opt_traj_length": 9, "opt_traj_reward": -1.0695166398401517, "test_difficulty": "high", "tag": 3, "all_opt_actions": [["left", "left", "down", "down", "right", "right", "down", "right", "down"]]}], [{"agent": {"x": 3, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 0, 1, 1, 1, 1, 1, 1, 0], "opt_actions": ["left", "left", "down", "down", "right", "right", "right", "right", "down", "down"], "opt_traj_length": 10, "opt_traj_reward": -1.188351822044613, "test_difficulty": "high", "tag": 4, "all_opt_actions": [["left", "left", "down", "down", "right", "right", "right", "right", "down", "down"]]}, {"agent": {"x": 3, "y": 5}, "goal": {"x": 5, "y": 1}, "walls": [], "A_tiles": [{"x": 2, "y": 2}, {"x": 4, "y": 2}], "available_A_tiles": [{"x": 2, "y": 2}, {"x": 3, "y": 2}, {"x": 4, "y": 2}, {"x": 4, "y": 1}], "B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 3}], "available_B_tiles": [{"x": 2, "y": 4}, {"x": 3, "y": 4}, {"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], "width": 5, "height": 5, "gamma": 1, "env_code": [1, 0, 1, 0, 1, 1, 1, 0, 1], "opt_actions": ["left", "left", "down", "down", "down", "down", "right", "right", "right", "right"], "opt_traj_length": 10, "opt_traj_reward": -1.188351822044613, "test_difficulty": "high", "tag": 5, "all_opt_actions": [["left", "left", "down", "down", "down", "down", "right", "right", "right", "right"], ["left", "left", "down", "down", "right", "right", "down", "down", "right", "right"]]}]]}, "skateboard2": {"low": [[{"agent": {"x": 6, "y": 3, "has_skateboard": 0}, "skateboard": [{"x": 2, "y": 3, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 1, 1], "opt_actions": ["up"], "opt_traj_length": 1, "opt_traj_reward": -0.8793332784666024, "test_difficulty": "low", "tag": 0, "all_opt_actions": [["up"]]}, {"agent": {"x": 2, "y": 3, "has_skateboard": 0}, "skateboard": [{"x": 6, "y": 1, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["up", "right", "right", "right", "right"], "opt_traj_length": 5, "opt_traj_reward": -4.396666392333012, "test_difficulty": "low", "tag": 1, "all_opt_actions": [["up", "right", "right", "right", "right"], ["right", "up", "right", "right", "right"], ["right", "right", "up", "right", "right"], ["right", "right", "right", "up", "right"], ["right", "right", "right", "right", "up"]]}], [{"agent": {"x": 6, "y": 3, "has_skateboard": 0}, "skateboard": [{"x": 5, "y": 1, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["up"], "opt_traj_length": 1, "opt_traj_reward": -0.8793332784666024, "test_difficulty": "low", "tag": 2, "all_opt_actions": [["up"]]}, {"agent": {"x": 3, "y": 3, "has_skateboard": 0}, "skateboard": [{"x": 4, "y": 1, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 0, 1], "opt_actions": ["up", "right", "right", "right"], "opt_traj_length": 4, "opt_traj_reward": -3.5173331138664095, "test_difficulty": "low", "tag": 3, "all_opt_actions": [["up", "right", "right", "right"], ["right", "up", "right", "right"], ["right", "right", "up", "right"], ["right", "right", "right", "up"]]}], [{"agent": {"x": 6, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 1, 1], "opt_actions": ["up", "up", "up"], "opt_traj_length": 3, "opt_traj_reward": -2.2093248621473385, "test_difficulty": "low", "tag": 4, "all_opt_actions": [["up", "up", "up"]]}, {"agent": {"x": 5, "y": 4, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 1, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 0, 1], "opt_actions": ["right"], "opt_traj_length": 1, "opt_traj_reward": -0.8793332784666024, "test_difficulty": "low", "tag": 5, "all_opt_actions": [["right"]]}]], "medium": [[{"agent": {"x": 1, "y": 2, "has_skateboard": 0}, "skateboard": [{"x": 2, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["up", "up", "right", "pickup", "right", "right", "right", "right"], "opt_traj_length": 8, "opt_traj_reward": -3.407416454058084, "test_difficulty": "medium", "tag": 0, "all_opt_actions": [["up", "up", "right", "pickup", "right", "right", "right", "right"], ["up", "right", "up", "pickup", "right", "right", "right", "right"], ["right", "up", "up", "pickup", "right", "right", "right", "right"]]}, {"agent": {"x": 1, "y": 4, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 1, 1], "opt_actions": ["pickup", "right", "right", "right", "right", "right"], "opt_traj_length": 6, "opt_traj_reward": -0.9232999423899325, "test_difficulty": "medium", "tag": 1, "all_opt_actions": [["pickup", "right", "right", "right", "right", "right"]]}], [{"agent": {"x": 1, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 2, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["up", "pickup", "up", "up", "right", "right", "right", "right", "right"], "opt_traj_length": 9, "opt_traj_reward": -1.681724895067377, "test_difficulty": "medium", "tag": 2, "all_opt_actions": [["up", "pickup", "up", "up", "right", "right", "right", "right", "right"], ["up", "pickup", "up", "right", "up", "right", "right", "right", "right"], ["up", "pickup", "up", "right", "right", "up", "right", "right", "right"], ["up", "pickup", "up", "right", "right", "right", "up", "right", "right"], ["up", "pickup", "up", "right", "right", "right", "right", "up", "right"], ["up", "pickup", "up", "right", "right", "right", "right", "right", "up"], ["up", "pickup", "right", "up", "up", "right", "right", "right", "right"], ["up", "pickup", "right", "up", "right", "up", "right", "right", "right"], ["up", "pickup", "right", "up", "right", "right", "up", "right", "right"], ["up", "pickup", "right", "up", "right", "right", "right", "up", "right"], ["up", "pickup", "right", "up", "right", "right", "right", "right", "up"]]}, {"agent": {"x": 3, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 3, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 0, 1], "opt_actions": ["up", "up", "up", "pickup", "right", "right", "right"], "opt_traj_length": 7, "opt_traj_reward": -3.2535331303264288, "test_difficulty": "medium", "tag": 3, "all_opt_actions": [["up", "up", "up", "pickup", "right", "right", "right"]]}], [{"agent": {"x": 3, "y": 2, "has_skateboard": 0}, "skateboard": [{"x": 6, "y": 2, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["right", "right", "right", "pickup", "up", "up"], "opt_traj_length": 6, "opt_traj_reward": -3.099649806594773, "test_difficulty": "medium", "tag": 4, "all_opt_actions": [["right", "right", "right", "pickup", "up", "up"]]}, {"agent": {"x": 4, "y": 2, "has_skateboard": 0}, "skateboard": [{"x": 4, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 0, 1], "opt_actions": ["up", "up", "pickup", "right", "right"], "opt_traj_length": 5, "opt_traj_reward": -2.220316528128171, "test_difficulty": "medium", "tag": 5, "all_opt_actions": [["up", "up", "pickup", "right", "right"]]}]], "high": [[{"agent": {"x": 2, "y": 4, "has_skateboard": 0}, "skateboard": [{"x": 2, "y": 2, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 1, 1], "opt_actions": ["down", "down", "pickup", "up", "up", "right", "right", "right", "right"], "opt_traj_length": 9, "opt_traj_reward": -2.835849823054793, "test_difficulty": "high", "tag": 0, "all_opt_actions": [["down", "down", "pickup", "up", "up", "right", "right", "right", "right"], ["down", "down", "pickup", "up", "right", "up", "right", "right", "right"], ["down", "down", "pickup", "up", "right", "right", "up", "right", "right"], ["down", "down", "pickup", "up", "right", "right", "right", "up", "right"], ["down", "down", "pickup", "up", "right", "right", "right", "right", "up"], ["down", "down", "pickup", "right", "up", "up", "right", "right", "right"], ["down", "down", "pickup", "right", "up", "right", "up", "right", "right"], ["down", "down", "pickup", "right", "up", "right", "right", "up", "right"], ["down", "down", "pickup", "right", "up", "right", "right", "right", "up"], ["down", "down", "pickup", "right", "right", "up", "up", "right", "right"], ["down", "down", "pickup", "right", "right", "up", "right", "up", "right"]]}, {"agent": {"x": 1, "y": 2, "has_skateboard": 0}, "skateboard": [{"x": 6, "y": 3, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 1, 1], "opt_actions": ["down", "right", "right", "right", "right", "right", "up", "up", "pickup", "up"], "opt_traj_length": 10, "opt_traj_reward": -4.770383035681318, "test_difficulty": "high", "tag": 1, "all_opt_actions": [["down", "right", "right", "right", "right", "right", "up", "up", "pickup", "up"]]}], [{"agent": {"x": 4, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 3, "y": 4, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 0, 1], "opt_actions": ["left", "up", "up", "up", "pickup", "right", "right", "right"], "opt_traj_length": 8, "opt_traj_reward": -3.2755164622880937, "test_difficulty": "high", "tag": 2, "all_opt_actions": [["left", "up", "up", "up", "pickup", "right", "right", "right"]]}, {"agent": {"x": 3, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 3, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [1, 1, 1], "opt_actions": ["left", "left", "up", "up", "pickup", "up", "right", "right", "right", "right", "right"], "opt_traj_length": 11, "opt_traj_reward": -3.3084914602305915, "test_difficulty": "high", "tag": 3, "all_opt_actions": [["left", "left", "up", "up", "pickup", "up", "right", "right", "right", "right", "right"], ["left", "left", "up", "up", "pickup", "right", "up", "right", "right", "right", "right"], ["left", "left", "up", "up", "pickup", "right", "right", "up", "right", "right", "right"], ["left", "left", "up", "up", "pickup", "right", "right", "right", "up", "right", "right"], ["left", "left", "up", "up", "pickup", "right", "right", "right", "right", "up", "right"], ["left", "left", "up", "up", "pickup", "right", "right", "right", "right", "right", "up"]]}], [{"agent": {"x": 5, "y": 1, "has_skateboard": 0}, "skateboard": [{"x": 3, "y": 2, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 1, 1], "opt_actions": ["left", "left", "up", "pickup", "up", "up", "right", "right", "right"], "opt_traj_length": 9, "opt_traj_reward": -2.2752748580323336, "test_difficulty": "high", "tag": 4, "all_opt_actions": [["left", "left", "up", "pickup", "up", "up", "right", "right", "right"], ["left", "left", "up", "pickup", "up", "right", "up", "right", "right"], ["left", "left", "up", "pickup", "up", "right", "right", "up", "right"], ["left", "left", "up", "pickup", "up", "right", "right", "right", "up"], ["left", "left", "up", "pickup", "right", "up", "up", "right", "right"], ["left", "left", "up", "pickup", "right", "up", "right", "up", "right"], ["left", "left", "up", "pickup", "right", "up", "right", "right", "up"], ["left", "left", "up", "pickup", "right", "right", "up", "up", "right"], ["left", "left", "up", "pickup", "right", "right", "up", "right", "up"], ["left", "left", "up", "pickup", "right", "right", "right", "up", "up"]]}, {"agent": {"x": 1, "y": 4, "has_skateboard": 0}, "skateboard": [{"x": 1, "y": 1, "on_agent": 0}], "goal": {"x": 6, "y": 4}, "walls": [], "available_paths": [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 2}, {"x": 6, "y": 3}], "paths": [{"x": 3, "y": 1}, {"x": 4, "y": 1}, {"x": 5, "y": 1}, {"x": 6, "y": 1}], "width": 6, "height": 4, "gamma": 1, "env_code": [0, 1, 1], "opt_actions": ["down", "down", "down", "pickup", "up", "up", "up", "right", "right", "right", "right", "right"], "opt_traj_length": 12, "opt_traj_reward": -4.022949748984706, "test_difficulty": "high", "tag": 5, "all_opt_actions": [["down", "down", "down", "pickup", "up", "up", "up", "right", "right", "right", "right", "right"], ["down", "down", "down", "pickup", "up", "up", "right", "up", "right", "right", "right", "right"], ["down", "down", "down", "pickup", "up", "up", "right", "right", "up", "right", "right", "right"], ["down", "down", "down", "pickup", "up", "up", "right", "right", "right", "up", "right", "right"], ["down", "down", "down", "pickup", "up", "up", "right", "right", "right", "right", "up", "right"], ["down", "down", "down", "pickup", "up", "up", "right", "right", "right", "right", "right", "up"], ["down", "down", "down", "pickup", "up", "right", "up", "up", "right", "right", "right", "right"], ["down", "down", "down", "pickup", "up", "right", "up", "right", "up", "right", "right", "right"], ["down", "down", "down", "pickup", "up", "right", "up", "right", "right", "up", "right", "right"], ["down", "down", "down", "pickup", "up", "right", "up", "right", "right", "right", "up", "right"], ["down", "down", "down", "pickup", "up", "right", "up", "right", "right", "right", "right", "up"]]}]]}}

    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['low'][train_test_set][0], 'augmented_taxi2', testing_simulation_header));
    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['low'][train_test_set][1], 'augmented_taxi2', testing_simulation_header));
    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['medium'][train_test_set][0], 'augmented_taxi2', testing_simulation_header));
    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['medium'][train_test_set][1], 'augmented_taxi2', testing_simulation_header));
    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['high'][train_test_set][0], 'augmented_taxi2', testing_simulation_header));
    augmented_taxi2_testing.push(testing_simulation("/static/external/augmented_taxi2.html", testing_mdp_parameters['augmented_taxi2']['high'][train_test_set][1], 'augmented_taxi2', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['low'][train_test_set][0], 'colored_tiles', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['low'][train_test_set][1], 'colored_tiles', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['medium'][train_test_set][0], 'colored_tiles', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['medium'][train_test_set][1], 'colored_tiles', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['high'][train_test_set][0], 'colored_tiles', testing_simulation_header));
    colored_tiles_testing.push(testing_simulation("/static/external/colored_tiles.html", testing_mdp_parameters['colored_tiles']['high'][train_test_set][1], 'colored_tiles', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['low'][train_test_set][0], 'skateboard2', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['low'][train_test_set][1], 'skateboard2', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['medium'][train_test_set][0], 'skateboard2', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['medium'][train_test_set][1], 'skateboard2', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['high'][train_test_set][0], 'skateboard2', testing_simulation_header));
    skateboard2_testing.push(testing_simulation("/static/external/skateboard2.html", testing_mdp_parameters['skateboard2']['high'][train_test_set][1], 'skateboard2', testing_simulation_header));

    // shuffle the testing arrays to mix between low, medium, and high test difficulties
    augmented_taxi2_testing = shuffle(augmented_taxi2_testing);
    colored_tiles_testing = shuffle(colored_tiles_testing);
    skateboard2_testing = shuffle(skateboard2_testing);

    switch (counterbalance) {
        case 0:
            order = ['augmented_taxi2', 'colored_tiles', 'skateboard2'];
            break;
        case 1:
            order = ['augmented_taxi2', 'skateboard2', 'colored_tiles'];
            break;
        case 2:
            order = ['colored_tiles', 'augmented_taxi2', 'skateboard2'];
            break;
        case 3:
            order = ['colored_tiles', 'skateboard2', 'augmented_taxi2'];
            break;
        case 4:
            order = ['skateboard2', 'augmented_taxi2', 'colored_tiles'];
            break;
        case 5:
            order = ['skateboard2', 'colored_tiles', 'augmented_taxi2'];
            break;
    }

    timeline = [];

    // intro pages
    timeline.push(instructions_intro);
    timeline.push(instructions_overview);

    // practice game
    timeline.push(sandbox_introduction);
    timeline.push(testing_simulation("/static/external/sandbox.html", sandbox_2_parameters, 'sandbox', sandbox_2_header, 'free_play'));
    timeline.push(testing_simulation("/static/external/sandbox.html", sandbox_1_parameters, 'sandbox', sandbox_3_header, 'optimal_traj_1'));
    timeline.push(post_practice_button);

    for (var i = 0; i < order.length; i++) {
        if (order[i] == 'augmented_taxi2') {
            timeline.push(augmented_taxi2_introduction);

            timeline.push(training_video(augmented_taxi2_videos, 'augmented_taxi2', training_video_header))

            timeline.push(training_survey('augmented_taxi2'));

            for (var k = 0; k < augmented_taxi2_testing.length; k++) {
                timeline.push(augmented_taxi2_testing[k]);
            }
        } else if (order[i] == 'colored_tiles') {
            timeline.push(colored_tiles_introduction);

            timeline.push(training_video(colored_tiles_videos, 'colored_tiles', training_video_header))

            timeline.push(training_survey('colored_tiles'));

            for (var k = 0; k < colored_tiles_testing.length; k++) {
                timeline.push(colored_tiles_testing[k]);
            }
        } else {
            timeline.push(skateboard2_introduction);

            timeline.push(training_video(skateboard2_videos, 'skateboard2', training_video_header))

            timeline.push(training_survey('skateboard2'));

            for (var k = 0; k < skateboard2_testing.length; k++) {
                timeline.push(skateboard2_testing[k]);
            }
        }
        if (i < order.length - 1) {
            timeline.push(post_main_game(post_main_game_header, order[i]));
        }
        else {
            timeline.push(post_main_game(post_main_game_header_final, order[i]));
        }
    }
    timeline.push(post_study_survey);
    timeline.push(debrief);

    jsPsych.init({
        timeline: timeline,
        show_progress_bar: true,
        display_element: 'jspsych-target',

        on_finish: function () {
            psiTurk.saveData({
                success: function() {
                    psiTurk.completeHIT();
                }
            });
        // jsPsych.data.displayData();
        },

        on_data_update:function(data) {
            psiTurk.recordTrialData(data)
            psiTurk.saveData()
        }
    });