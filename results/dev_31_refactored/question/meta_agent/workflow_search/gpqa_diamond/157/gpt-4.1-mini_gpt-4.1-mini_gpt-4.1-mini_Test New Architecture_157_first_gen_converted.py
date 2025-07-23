async def forward_157(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and summarize key information about the transcription factor, mutations X and Y, and their functional domains. "
        "Input: taskInfo containing the question and choices."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Analyze functional relationships between phosphorylation, dimerization, mutation effects, and dominant-negative mechanisms. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_1."
    )
    cot_agent_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_agent_desc_0_2
    )
    logs.append(log_0_2)

    loop_results = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []},
        "stage_1.subtask_3": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_1_1 = (
            "Sub-task 1: Generate initial hypotheses on molecular consequences of mutation Y based on dominant-negative mutation principles. "
            "Input: taskInfo, and all previous thinking and answers from stage_1.subtask_2 and stage_1.subtask_3 if any."
        )
        inputs_1_1 = [taskInfo, results_0_2['thinking'], results_0_2['answer']]
        if iteration > 0:
            inputs_1_1 += loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"]
            inputs_1_1 += loop_results["stage_1.subtask_3"]["thinking"] + loop_results["stage_1.subtask_3"]["answer"]

        final_decision_instruction_1_1 = (
            "Sub-task 1: Synthesize and choose the most consistent initial hypotheses on molecular consequences of mutation Y."
        )

        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": inputs_1_1,
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] +
                            (loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"] if iteration > 0 else []) +
                            (loop_results["stage_1.subtask_3"]["thinking"] + loop_results["stage_1.subtask_3"]["answer"] if iteration > 0 else [])
        }

        results_1_1, log_1_1 = await self.sc_cot(
            subtask_id=f"stage_1.subtask_1.iter{iteration+1}",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=self.max_sc
        )
        logs.append(log_1_1)

        loop_results["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_sc_instruction_1_2 = (
            "Sub-task 2: Refine hypotheses by evaluating each phenotype option against known dominant-negative mutation effects and query context. "
            "Input: taskInfo, and all previous thinking and answers from stage_1.subtask_1."
        )
        inputs_1_2 = [taskInfo] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"]

        final_decision_instruction_1_2 = (
            "Sub-task 2: Synthesize and choose the most consistent refined hypotheses on molecular phenotype of mutation Y."
        )

        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": inputs_1_2,
            "temperature": 0.5,
            "context_desc": ["user query"] + loop_results["stage_1.subtask_1"]["thinking"] + loop_results["stage_1.subtask_1"]["answer"]
        }

        results_1_2, log_1_2 = await self.sc_cot(
            subtask_id=f"stage_1.subtask_2.iter{iteration+1}",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=self.max_sc
        )
        logs.append(log_1_2)

        loop_results["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

        cot_sc_instruction_1_3 = (
            "Sub-task 3: Validate refined hypotheses for consistency with recessive vs dominant mutation behavior and molecular biology principles. "
            "Input: taskInfo, and all previous thinking and answers from stage_1.subtask_2."
        )
        inputs_1_3 = [taskInfo] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"]

        final_decision_instruction_1_3 = (
            "Sub-task 3: Synthesize and choose the most consistent validated hypotheses on molecular phenotype of mutation Y."
        )

        cot_sc_desc_1_3 = {
            "instruction": cot_sc_instruction_1_3,
            "final_decision_instruction": final_decision_instruction_1_3,
            "input": inputs_1_3,
            "temperature": 0.5,
            "context_desc": ["user query"] + loop_results["stage_1.subtask_2"]["thinking"] + loop_results["stage_1.subtask_2"]["answer"]
        }

        results_1_3, log_1_3 = await self.sc_cot(
            subtask_id=f"stage_1.subtask_3.iter{iteration+1}",
            cot_agent_desc=cot_sc_desc_1_3,
            n_repeat=self.max_sc
        )
        logs.append(log_1_3)

        loop_results["stage_1.subtask_3"]["thinking"].append(results_1_3["thinking"])
        loop_results["stage_1.subtask_3"]["answer"].append(results_1_3["answer"])

    cot_agent_instruction_2_1 = (
        "Sub-task 1: Evaluate all candidate phenotypes against refined reasoning outputs and select the most plausible molecular phenotype. "
        "Input: taskInfo, thinking and answer from stage_0.subtask_2, and all thinking and answers from stage_1.subtask_3 iterations."
    )
    inputs_2_1 = [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results["stage_1.subtask_3"]["thinking"] + loop_results["stage_1.subtask_3"]["answer"]

    cot_agent_desc_2_1 = {
        "instruction": cot_agent_instruction_2_1,
        "input": inputs_2_1,
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + loop_results["stage_1.subtask_3"]["thinking"] + loop_results["stage_1.subtask_3"]["answer"]
    }

    results_2_1, log_2_1 = await self.answer_generate(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
