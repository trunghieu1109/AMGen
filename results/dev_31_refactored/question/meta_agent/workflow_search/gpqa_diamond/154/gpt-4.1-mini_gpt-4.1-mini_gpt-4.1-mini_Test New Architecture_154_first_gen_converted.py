async def forward_154(self, taskInfo):
    logs = []
    results = {}

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and verify the given operator matrices Px, Py, Pz and the state vector in the Pz eigenbasis. "
        "Input: the query containing matrix components and the state vector as described."
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
    results["stage_0.subtask_1"] = results_0_1

    cot_instruction_1_1 = (
        "Sub-task 1: Calculate the expectation value <Pz> using the given state vector and Pz matrix. "
        "Input: results from stage_0.subtask_1 (thinking and answer)."
    )
    cot_agent_desc_1_1 = {
        "instruction": cot_instruction_1_1,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_1, log_1_1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc_1_1
    )
    logs.append(log_1_1)
    results["stage_1.subtask_1"] = results_1_1

    cot_instruction_1_2 = (
        "Sub-task 2: Calculate the expectation value <Pz^2> using the given state vector and Pz matrix. "
        "Input: results from stage_0.subtask_1 (thinking and answer)."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_0_1["thinking"], results_0_1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)
    results["stage_1.subtask_2"] = results_1_2

    reflect_instruction_2_1 = (
        "Sub-task 1: Compute the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) using the expectation values. "
        "Input: results from stage_1.subtask_1 and stage_1.subtask_2 (thinking and answer)."
    )
    critic_instruction_2_1 = (
        "Please review and provide the limitations of the provided solution for calculating ΔPz, "
        "checking for correctness and completeness."
    )
    reflect_desc_2_1 = {
        "instruction": reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"], results_1_2["thinking"], results_1_2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results_2_1, log_2_1 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=reflect_desc_2_1,
        n_repeat=1
    )
    logs.append(log_2_1)
    results["stage_2.subtask_1"] = results_2_1

    loop_results = {"stage_3.subtask_1": {"thinking": [], "answer": []}}
    for i in range(2):
        reflect_instruction_3_1 = (
            f"Iteration {i+1}: Generate intermediate reasoning steps to verify and refine the calculation of ΔPz. "
            "Input: results from stage_2.subtask_1 (thinking and answer) and all previous iterations of this subtask."
        )
        critic_instruction_3_1 = (
            f"Iteration {i+1}: Please review and critique the intermediate solution for ΔPz, "
            "highlighting any errors or improvements."
        )
        inputs_for_3_1 = [taskInfo, results_2_1["thinking"], results_2_1["answer"]] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"]
        reflect_desc_3_1 = {
            "instruction": reflect_instruction_3_1,
            "critic_instruction": critic_instruction_3_1,
            "input": inputs_for_3_1,
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"] + ["thinking of previous iteration"]*len(loop_results["stage_3.subtask_1"]["thinking"]) + ["answer of previous iteration"]*len(loop_results["stage_3.subtask_1"]["answer"])
        }
        results_3_1, log_3_1 = await self.reflexion(
            subtask_id="stage_3.subtask_1",
            reflect_desc=reflect_desc_3_1,
            n_repeat=1
        )
        logs.append(log_3_1)
        loop_results["stage_3.subtask_1"]["thinking"].append(results_3_1["thinking"])
        loop_results["stage_3.subtask_1"]["answer"].append(results_3_1["answer"])

    debate_instruction_4_1 = (
        "Sub-task 1: Evaluate candidate uncertainty values and select the answer that best matches the computed ΔPz. "
        "Input: results from all iterations of stage_3.subtask_1 (thinking and answer)."
    )
    final_decision_instruction_4_1 = (
        "Sub-task 1: Select the best answer for ΔPz from the given choices based on the refined calculations and reasoning."
    )
    debate_desc_4_1 = {
        "instruction": debate_instruction_4_1,
        "final_decision_instruction": final_decision_instruction_4_1,
        "input": [taskInfo] + loop_results["stage_3.subtask_1"]["thinking"] + loop_results["stage_3.subtask_1"]["answer"],
        "context_desc": ["user query"] + ["thinking of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["thinking"]) + ["answer of stage_3.subtask_1"]*len(loop_results["stage_3.subtask_1"]["answer"]),
        "temperature": 0.5
    }
    results_4_1, log_4_1 = await self.debate(
        subtask_id="stage_4.subtask_1",
        debate_desc=debate_desc_4_1,
        n_repeat=1
    )
    logs.append(log_4_1)
    results["stage_4.subtask_1"] = results_4_1

    final_answer = await self.make_final_answer(results_4_1["thinking"], results_4_1["answer"])
    return final_answer, logs
