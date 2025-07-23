async def forward_173(self, taskInfo):
    logs = []
    loop_results_stage1 = {"stage_1.subtask_1": {"thinking": [], "answer": []}, "stage_1.subtask_2": {"thinking": [], "answer": []}}

    cot_sc_instruction_0_1 = (
        "Sub-task 1: Extract given data and define physical parameters: initial mass M, fragment mass ratio, rest mass sum, and energy relations. "
        "Input: user query containing problem statement."
    )
    cot_sc_final_decision_0_1 = (
        "Sub-task 1: Synthesize and choose the most consistent extraction and parameter setup for the problem."
    )
    cot_sc_desc_0_1 = {
        "instruction": cot_sc_instruction_0_1,
        "final_decision_instruction": cot_sc_final_decision_0_1,
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_0_1, log_0_1 = await self.sc_cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_sc_desc_0_1, n_repeat=3)
    logs.append(log_0_1)

    cot_instruction_0_2 = (
        "Sub-task 2: Formulate conservation laws and initial equations for relativistic and classical kinetic energy calculations. "
        "Input: results (thinking and answer) from stage_0.subtask_1."
    )
    cot_desc_0_2 = {
        "instruction": cot_instruction_0_2,
        "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_0_2, log_0_2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_desc_0_2)
    logs.append(log_0_2)

    for iteration in range(2):
        cot_instruction_1_1 = (
            f"Iteration {iteration+1} - Sub-task 1: Calculate relativistic kinetic energy T1 of the more massive fragment using conservation of momentum and energy. "
            "Input: results from stage_0.subtask_2 and all previous iterations of stage_1.subtask_2."
        )
        cot_desc_1_1 = {
            "instruction": cot_instruction_1_1,
            "input": [taskInfo, results_0_2['thinking'], results_0_2['answer']] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"] + ["thinking of previous stage_1.subtask_2 iterations"]*len(loop_results_stage1["stage_1.subtask_2"]["thinking"])
        }
        results_1_1, log_1_1 = await self.cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_desc_1_1)
        logs.append(log_1_1)
        loop_results_stage1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])

        cot_agent_instruction_1_2 = (
            f"Iteration {iteration+1} - Sub-task 2: Calculate classical (non-relativistic) kinetic energy T1 approximation and compare with relativistic T1. "
            "Input: results from stage_1.subtask_1 and stage_0.subtask_2."
        )
        cot_agent_desc_1_2 = {
            "instruction": cot_agent_instruction_1_2,
            "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_0_2["thinking"], results_0_2["answer"]],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results_1_2, log_1_2 = await self.answer_generate(subtask_id="stage_1.subtask_2", cot_agent_desc=cot_agent_desc_1_2)
        logs.append(log_1_2)
        loop_results_stage1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Compute the difference between relativistic and classical T1 values and evaluate candidate numerical answers. "
        "Input: results from all iterations of stage_1.subtask_2."
    )
    cot_sc_final_decision_2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent difference value between relativistic and classical T1, matching given choices."
    )
    cot_sc_desc_2_1 = {
        "instruction": cot_sc_instruction_2_1,
        "final_decision_instruction": cot_sc_final_decision_2_1,
        "input": [taskInfo] + loop_results_stage1["stage_1.subtask_2"]["thinking"] + loop_results_stage1["stage_1.subtask_2"]["answer"],
        "temperature": 0.5,
        "context_desc": ["user query"] + ["thinking of stage_1.subtask_2 iterations"]*len(loop_results_stage1["stage_1.subtask_2"]["thinking"]) + ["answer of stage_1.subtask_2 iterations"]*len(loop_results_stage1["stage_1.subtask_2"]["answer"])
    }
    results_2_1, log_2_1 = await self.sc_cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_sc_desc_2_1, n_repeat=3)
    logs.append(log_2_1)

    cot_agent_instruction_3_1 = (
        "Sub-task 1: Select the correct numerical difference from given choices and validate consistency with physics principles. "
        "Input: results from stage_2.subtask_1."
    )
    cot_agent_desc_3_1 = {
        "instruction": cot_agent_instruction_3_1,
        "input": [taskInfo, results_2_1["thinking"], results_2_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_3_1, log_3_1 = await self.answer_generate(subtask_id="stage_3.subtask_1", cot_agent_desc=cot_agent_desc_3_1)
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
