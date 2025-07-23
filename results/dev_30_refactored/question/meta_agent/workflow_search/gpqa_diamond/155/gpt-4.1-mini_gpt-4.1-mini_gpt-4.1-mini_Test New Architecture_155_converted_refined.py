async def forward_155(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []},
        'stage_0.subtask_6': {'thinking': [], 'answer': []},
        'stage_0.subtask_7': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_01 = (
            "Sub-task 1: Analyze the stereochemical outcome of Reaction 1: epoxidation of (E)-oct-4-ene with mCPBA, "
            "identifying the stereoisomers formed immediately after epoxidation. Explicitly note that this step excludes the aqueous acid workup to avoid the previous error of assuming stable epoxides. "
            "Input content: taskInfo"
        )
        cot_agent_desc_01 = {
            'instruction': cot_instruction_01,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_01, log_01 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_01
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_01['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_01['answer'])
        logs.append(log_01)

        cot_sc_instruction_02 = (
            "Sub-task 2: Analyze the stereochemical outcome of Reaction 1 after aqueous acid treatment: determine the stereochemical changes due to acid-catalyzed ring-opening of the epoxide to vicinal diols, "
            "listing the resulting diol stereoisomers. This subtask addresses the previous failure to consider ring-opening and its stereochemical consequences. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and previous iterations of stage_0.subtask_1"
        )
        final_decision_instruction_02 = (
            "Sub-task 2: Synthesize and choose the most consistent answer for the stereochemical outcome of Reaction 1 after aqueous acid treatment."
        )
        cot_sc_desc_02 = {
            'instruction': cot_sc_instruction_02,
            'final_decision_instruction': final_decision_instruction_02,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_02, log_02 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_sc_desc_02,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_02['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_02['answer'])
        logs.append(log_02)

        cot_instruction_03 = (
            "Sub-task 3: Analyze the stereochemical outcome of Reaction 2: epoxidation of (Z)-oct-4-ene with mCPBA, identifying the stereoisomers formed immediately after epoxidation, "
            "excluding aqueous acid workup effects to maintain clarity and avoid prior mistakes. "
            "Input content: results (thinking and answer) from stage_0.subtask_1 and previous iterations of stage_0.subtask_1"
        )
        cot_agent_desc_03 = {
            'instruction': cot_instruction_03,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_03, log_03 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_03
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_03['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_03['answer'])
        logs.append(log_03)

        cot_sc_instruction_04 = (
            "Sub-task 4: Analyze the stereochemical outcome of Reaction 2 after aqueous acid treatment: determine the stereochemical changes due to acid-catalyzed ring-opening of the epoxide to vicinal diols, "
            "listing the resulting diol stereoisomers. This explicitly incorporates the previously missing ring-opening step. "
            "Input content: results (thinking and answer) from stage_0.subtask_3, stage_0.subtask_2, and previous iterations of stage_0.subtask_3"
        )
        final_decision_instruction_04 = (
            "Sub-task 4: Synthesize and choose the most consistent answer for the stereochemical outcome of Reaction 2 after aqueous acid treatment."
        )
        cot_sc_desc_04 = {
            'instruction': cot_sc_instruction_04,
            'final_decision_instruction': final_decision_instruction_04,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_04, log_04 = await self.sc_cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_sc_desc_04,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_04['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_04['answer'])
        logs.append(log_04)

        cot_instruction_05 = (
            "Sub-task 5: Determine the stereochemical relationship between the diol products of Reaction 1 and Reaction 2, classifying them as enantiomers, diastereomers, or meso forms. "
            "Use the outputs from both ring-opening analyses to avoid the prior error of comparing epoxides instead of diols. "
            "Input content: results (thinking and answer) from stage_0.subtask_2, stage_0.subtask_4, and previous iterations of stage_0.subtask_4"
        )
        cot_agent_desc_05 = {
            'instruction': cot_instruction_05,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_05, log_05 = await self.cot(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_05
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_05['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_05['answer'])
        logs.append(log_05)

        cot_instruction_06 = (
            "Sub-task 6: Predict the total number of stereoisomers present in the combined product mixture after aqueous acid treatment, based on the stereochemical relationships established. "
            "This subtask replaces the previous single prediction subtask with a more explicit focus on diol stereoisomers. "
            "Input content: results (thinking and answer) from stage_0.subtask_5 and previous iterations of stage_0.subtask_5"
        )
        cot_agent_desc_06 = {
            'instruction': cot_instruction_06,
            'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_06, log_06 = await self.cot(
            subtask_id='stage_0.subtask_6',
            cot_agent_desc=cot_agent_desc_06
        )
        loop_results['stage_0.subtask_6']['thinking'].append(results_06['thinking'])
        loop_results['stage_0.subtask_6']['answer'].append(results_06['answer'])
        logs.append(log_06)

        cot_agent_instruction_07 = (
            "Sub-task 7: Summarize the stereochemical composition of the combined product mixture, emphasizing the nature of stereoisomers relevant for chromatographic separation (e.g., presence of enantiomeric pairs, meso forms, and diastereomers). "
            "This summary must explicitly consider the ring-opened diol products to avoid previous reasoning errors. "
            "Input content: results (thinking and answer) from stage_0.subtask_6 and previous iterations of stage_0.subtask_6"
        )
        cot_agent_desc_07 = {
            'instruction': cot_agent_instruction_07,
            'input': [taskInfo] + loop_results['stage_0.subtask_6']['thinking'] + loop_results['stage_0.subtask_6']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_6', 'answer of stage_0.subtask_6']
        }
        results_07, log_07 = await self.answer_generate(
            subtask_id='stage_0.subtask_7',
            cot_agent_desc=cot_agent_desc_07
        )
        loop_results['stage_0.subtask_7']['thinking'].append(results_07['thinking'])
        loop_results['stage_0.subtask_7']['answer'].append(results_07['answer'])
        logs.append(log_07)

    aggregate_instruction_11 = (
        "Sub-task 1: Combine the stereochemical analyses from stage_0 to form a comprehensive description of the product mixture's stereoisomeric composition after aqueous acid treatment, "
        "ensuring all relevant stereochemical details are integrated. Input content: results (thinking and answer) from stage_0.subtask_7"
    )
    aggregate_desc_11 = {
        'instruction': aggregate_instruction_11,
        'input': [taskInfo] + loop_results['stage_0.subtask_7']['thinking'] + loop_results['stage_0.subtask_7']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_0.subtask_7']
    }
    results_11, log_11 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_11
    )
    logs.append(log_11)

    cot_instruction_12 = (
        "Sub-task 2: Integrate knowledge of chromatographic principles (achiral and chiral HPLC) with the stereochemical composition of the diol mixture to predict separation behavior, "
        "explicitly considering that achiral HPLC cannot separate enantiomers but can separate diastereomers, and chiral HPLC can separate both. "
        "Input content: results (thinking and answer) from stage_1.subtask_1"
    )
    cot_agent_desc_12 = {
        'instruction': cot_instruction_12,
        'input': [taskInfo, results_11['thinking'], results_11['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_12, log_12 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_12
    )
    logs.append(log_12)

    cot_instruction_13 = (
        "Sub-task 3: Identify which stereoisomers in the diol mixture are expected to be resolved by achiral HPLC and which require chiral HPLC for separation, "
        "explicitly addressing the previous confusion about peak counts and ensuring clarity on enantiomeric vs diastereomeric resolution. "
        "Input content: results (thinking and answer) from stage_1.subtask_2"
    )
    cot_agent_desc_13 = {
        'instruction': cot_instruction_13,
        'input': [taskInfo, results_12['thinking'], results_12['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_13, log_13 = await self.cot(
        subtask_id='stage_1.subtask_3',
        cot_agent_desc=cot_agent_desc_13
    )
    logs.append(log_13)

    cot_sc_instruction_21 = (
        "Sub-task 1: Select the number of distinct peaks expected in the standard (achiral) reverse-phase HPLC chromatogram based on the stereoisomeric differences of the diol products, "
        "ensuring the count reflects that enantiomers co-elute in achiral HPLC. Input content: results (thinking and answer) from stage_1.subtask_3"
    )
    final_decision_instruction_21 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the number of peaks in achiral HPLC chromatogram."
    )
    cot_sc_desc_21 = {
        'instruction': cot_sc_instruction_21,
        'final_decision_instruction': final_decision_instruction_21,
        'input': [taskInfo, results_13['thinking'], results_13['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_21, log_21 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_21,
        n_repeat=self.max_sc
    )
    logs.append(log_21)

    cot_sc_instruction_22 = (
        "Sub-task 2: Select the number of distinct peaks expected in the chiral HPLC chromatogram based on the ability to resolve both enantiomers and diastereomers in the diol mixture, "
        "ensuring consistency with stereochemical analysis. Input content: results (thinking and answer) from stage_1.subtask_3"
    )
    final_decision_instruction_22 = (
        "Sub-task 2: Synthesize and choose the most consistent answer for the number of peaks in chiral HPLC chromatogram."
    )
    cot_sc_desc_22 = {
        'instruction': cot_sc_instruction_22,
        'final_decision_instruction': final_decision_instruction_22,
        'input': [taskInfo, results_13['thinking'], results_13['answer']],
        'temperature': 0.5,
        'context': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_22, log_22 = await self.sc_cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_sc_desc_22,
        n_repeat=self.max_sc
    )
    logs.append(log_22)

    cot_instruction_23 = (
        "Sub-task 3: Compare and contrast the expected peak counts from both chromatographic methods to identify differences in resolution capabilities, "
        "explicitly highlighting why chiral HPLC shows more peaks than achiral HPLC in this context. Input content: results (thinking and answer) from stage_2.subtask_1 and stage_2.subtask_2"
    )
    cot_agent_desc_23 = {
        'instruction': cot_instruction_23,
        'input': [taskInfo, results_21['thinking'], results_21['answer'], results_22['thinking'], results_22['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_23, log_23 = await self.cot(
        subtask_id='stage_2.subtask_3',
        cot_agent_desc=cot_agent_desc_23
    )
    logs.append(log_23)

    review_instruction_31 = (
        "Sub-task 1: Evaluate the consistency of the predicted chromatographic peak counts with known stereochemical and chromatographic principles, "
        "ensuring no contradictions or overlooked factors remain. Input content: results (thinking and answer) from stage_2.subtask_3"
    )
    review_desc_31 = {
        'instruction': review_instruction_31,
        'input': [taskInfo, results_23['thinking'], results_23['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_3', 'answer of stage_2.subtask_3']
    }
    results_31, log_31 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_31
    )
    logs.append(log_31)

    cot_instruction_32 = (
        "Sub-task 2: Assess whether the predicted chromatograms align with the theoretical maximum chromatographic resolution assumption, "
        "confirming that the predicted number of peaks is the maximum possible under ideal conditions. Input content: results (thinking and answer) from stage_3.subtask_1"
    )
    cot_agent_desc_32 = {
        'instruction': cot_instruction_32,
        'input': [taskInfo, results_31['thinking'], results_31['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_32, log_32 = await self.cot(
        subtask_id='stage_3.subtask_2',
        cot_agent_desc=cot_agent_desc_32
    )
    logs.append(log_32)

    cot_reflect_instruction_41 = (
        "Sub-task 1: Refine the chromatographic predictions into a clear, concise summary suitable for final answer selection, "
        "explicitly incorporating the ring-opening step and chromatographic resolution principles to avoid previous errors. Input content: results (thinking and answer) from stage_3.subtask_2"
    )
    cot_reflect_desc_41 = {
        'instruction': cot_reflect_instruction_41,
        'input': [taskInfo, results_32['thinking'], results_32['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_3.subtask_2', 'answer of stage_3.subtask_2']
    }
    results_41, log_41 = await self.reflexion(
        subtask_id='stage_4.subtask_1',
        reflect_desc=cot_reflect_desc_41,
        n_repeat=self.max_round
    )
    logs.append(log_41)

    formatter_instruction_42 = (
        "Sub-task 2: Format the final answer by carefully matching the predicted chromatographic observations to one of the provided multiple-choice options, "
        "including an explicit validation step to cross-check the chemical reasoning against the exact wording of the choices to prevent mismatches as occurred previously. "
        "Input content: results (thinking and answer) from stage_4.subtask_1"
    )
    formatter_desc_42 = {
        'instruction': formatter_instruction_42,
        'input': [taskInfo, results_41['thinking'], results_41['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1'],
        'format': 'short and concise, without explanation'
    }
    results_42, log_42 = await self.specific_format(
        subtask_id='stage_4.subtask_2',
        formatter_desc=formatter_desc_42
    )
    logs.append(log_42)

    debate_instruction_43 = (
        "Sub-task 3: Conduct a final debate or reflexion step focused solely on validating the final answer choice, "
        "ensuring consistency between the chemical reasoning and the selected multiple-choice option, thereby preventing accidental selection errors. "
        "Input content: results (thinking and answer) from stage_4.subtask_2"
    )
    final_decision_instruction_43 = (
        "Sub-task 3: Validate and finalize the answer choice consistency with chemical reasoning."
    )
    debate_desc_43 = {
        'instruction': debate_instruction_43,
        'final_decision_instruction': final_decision_instruction_43,
        'input': [taskInfo, results_42['thinking'], results_42['answer']],
        'context': ['user query', 'thinking of stage_4.subtask_2', 'answer of stage_4.subtask_2'],
        'temperature': 0.5
    }
    results_43, log_43 = await self.debate(
        subtask_id='stage_4.subtask_3',
        debate_desc=debate_desc_43,
        n_repeat=self.max_round
    )
    logs.append(log_43)

    final_answer = await self.make_final_answer(results_43['thinking'], results_43['answer'])
    return final_answer, logs
