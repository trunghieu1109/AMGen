async def forward_157(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize key molecular and genetic information from the query, "
        "including the roles of phosphorylation, dimerization, and the nature of mutations X and Y."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_instruction2 = (
        "Sub-task 2: Identify and characterize the functional domains involved (transactivation and dimerization domains) "
        "and their roles in transcription factor activation and function, based on output from Sub-task 1."
    )
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_instruction3 = (
        "Sub-task 3: Define the molecular consequences of recessive loss-of-function versus dominant-negative mutations, "
        "focusing on how mutation Y in the dimerization domain can interfere with wild-type protein function, "
        "based on outputs from Sub-tasks 1 and 2."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of subtask 1",
            "answer of subtask 1",
            "thinking of subtask 2",
            "answer of subtask 2"
        ]
    }
    results3, log3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3,
        n_repeat=self.max_sc
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Assess the possible molecular phenotypes caused by mutation Y based on its dominant-negative nature, "
        "evaluating options such as loss of dimerization, aggregation, degradation, or conformational changes and their functional outcomes, "
        "based on output from Sub-task 3."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': [
            "user query",
            "thinking of subtask 3",
            "answer of subtask 3"
        ],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    debate_instruction5 = (
        "Sub-task 5: Integrate the analysis to select the most likely molecular phenotype observed in the presence of mutation Y from the given choices, "
        "justifying the selection based on molecular biology principles and dominant-negative mutation behavior, "
        "based on output from Sub-task 4."
    )
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': [
            "user query",
            "thinking of subtask 4",
            "answer of subtask 4"
        ],
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results5, log5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        n_repeat=self.max_round
    )
    logs.append(log5)

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'])
    return final_answer, logs
