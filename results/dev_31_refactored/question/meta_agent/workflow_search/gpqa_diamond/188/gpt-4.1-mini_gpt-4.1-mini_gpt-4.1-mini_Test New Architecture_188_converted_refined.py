async def forward_188(self, taskInfo):
    logs = []

    cot_instruction_0_1 = (
        "Sub-task 1: Identify and extract the list of effective particles and the key condition about spontaneous symmetry breaking from the query, "
        "ensuring clarity on the problem scope and key terms. Input content: taskInfo (query with question and choices)."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results_0_1, log_0_1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_0_1)
    logs.append(log_0_1)

    cot_instruction_1_1 = (
        "Sub-task 1: For each particle (Magnon, Skyrmion, Pion, Phonon), explicitly identify the continuous symmetry that is spontaneously broken "
        "and describe the resulting vacuum manifold (order parameter space). This step uses results from stage_0.subtask_1 (thinking and answer). "
        "Input content: taskInfo, results_0_1['thinking'], results_0_1['answer']"
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Using the symmetry and vacuum manifold analysis from stage_1.subtask_1, classify each particle as associated or not associated with spontaneously-broken continuous symmetry. "
        "Employ a self-critical chain-of-thought (SC-CoT) approach to verify and challenge the initial classification, explicitly checking if the particle's existence requires spontaneous symmetry breaking. "
        "Input content: taskInfo, results_1_1['thinking'], results_1_1['answer']"
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent classification for each particle regarding association with spontaneously-broken continuous symmetry."
    )
    cot_sc_desc_2_1 = {
        "instruction": cot_sc_instruction_2_1,
        "final_decision_instruction": final_decision_instruction_2_1,
        "input": [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.sc_cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_sc_desc_2_1, n_repeat=self.max_sc)
    logs.append(log_2_1)

    loop_results = {"stage_3.subtask_1": {"thinking": [], "answer": []}}
    for i in range(2):
        cot_instruction_3_1 = (
            f"Iteration {i+1} - Sub-task 1: Iteratively refine the classification and reasoning about each particle's connection to spontaneous symmetry breaking, "
            "using insights from prior iterations and the self-critical classification from stage_2.subtask_1. "
            "Input content: taskInfo, all previous thinking and answers from stage_2.subtask_1 and previous iterations of this subtask."
        )
        input_for_3_1 = [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"] + [results_2_1['thinking'], results_2_1['answer']]
        cot_agent_desc_3_1 = {
            "instruction": cot_instruction_3_1,
            "input": input_for_3_1,
            "temperature": 0.0,
            "context": ["user query"] + ["thinking of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["thinking"]) + ["answer of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["answer"]) + ["thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
        }
        results_3_1, log_3_1 = await self.cot(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
        logs.append(log_3_1)
        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])

    cot_instruction_4_1 = (
        "Sub-task 1: Combine the refined reasoning outcomes from iterative refinement (stage_3.subtask_1) to form a consolidated list indicating which particles are associated or not associated with spontaneously-broken continuous symmetry. "
        "Ensure the consolidation reflects the corrected understanding of skyrmions and other particles. "
        "Input content: taskInfo, all thinking and answers from stage_3.subtask_1 iterations."
    )
    cot_agent_desc_4_1 = {
        "instruction": cot_instruction_4_1,
        "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"],
        "temperature": 0.0,
        "context": ["user query"] + ["thinking of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["thinking"]) + ["answer of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["answer"])
    }
    results_4_1, log_4_1 = await self.cot(subtask_id="stage_4.subtask_1", cot_agent_desc=cot_agent_desc_4_1)
    logs.append(log_4_1)

    cot_instruction_5_1 = (
        "Sub-task 1: Evaluate the consolidated classification list to select the particle that is not associated with spontaneously-broken continuous symmetry. "
        "Justify the selection based on the detailed symmetry and vacuum manifold analysis and the refined classification. "
        "Input content: taskInfo, results_4_1['thinking'], results_4_1['answer']"
    )
    cot_agent_desc_5_1 = {
        "instruction": cot_instruction_5_1,
        "input": [taskInfo, results_4_1['thinking'], results_4_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_4.subtask_1", "answer of stage_4.subtask_1"]
    }
    results_5_1, log_5_1 = await self.cot(subtask_id="stage_5.subtask_1", cot_agent_desc=cot_agent_desc_5_1)
    logs.append(log_5_1)

    cot_agent_instruction_6_1 = (
        "Sub-task 1: Format the selected answer into a clear, concise final output suitable for direct response, ensuring no ambiguity remains. "
        "Input content: taskInfo, results_5_1['thinking'], results_5_1['answer']"
    )
    cot_agent_desc_6_1 = {
        "instruction": cot_agent_instruction_6_1,
        "input": [taskInfo, results_5_1['thinking'], results_5_1['answer']],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_5.subtask_1", "answer of stage_5.subtask_1"]
    }
    results_6_1, log_6_1 = await self.answer_generate(subtask_id="stage_6.subtask_1", cot_agent_desc=cot_agent_desc_6_1)
    logs.append(log_6_1)

    final_answer = await self.make_final_answer(results_6_1['thinking'], results_6_1['answer'])
    return final_answer, logs
