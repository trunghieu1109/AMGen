async def forward_171(self, taskInfo):
    logs = []

    results = {}

    cot_instruction_0_1 = (
        "Sub-task 1: Extract and quantify the given physical parameters: excitation ratio, energy difference, and assumptions (LTE, Boltzmann constant). "
        "Input: user query from taskInfo."
    )
    cot_agent_desc_0_1 = {
        "instruction": cot_instruction_0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context_desc": ["user query"]
    }
    results["stage_0.subtask_1"], log0_1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc_0_1
    )
    logs.append(log0_1)

    loop_results_stage_1 = {
        "stage_1.subtask_1": {"thinking": [], "answer": []},
        "stage_1.subtask_2": {"thinking": [], "answer": []}
    }

    for iteration in range(2):
        cot_sc_instruction_1_1 = (
            "Sub-task 1: Formulate the Boltzmann distribution relation for excitation ratio between star_1 and star_2 using ΔE, k, T_1, and T_2. "
            "Input: user query and outputs (thinking and answer) from stage_0.subtask_1."
        )
        final_decision_instruction_1_1 = (
            "Sub-task 1: Synthesize and choose the most consistent Boltzmann relation expression for excitation ratio."
        )
        cot_sc_desc_1_1 = {
            "instruction": cot_sc_instruction_1_1,
            "final_decision_instruction": final_decision_instruction_1_1,
            "input": [taskInfo, results["stage_0.subtask_1"]["thinking"], results["stage_0.subtask_1"]["answer"]] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_1_1, log1_1 = await self.sc_cot(
            subtask_id="stage_1.subtask_1",
            cot_agent_desc=cot_sc_desc_1_1,
            n_repeat=self.max_sc if hasattr(self, 'max_sc') else 3
        )
        loop_results_stage_1["stage_1.subtask_1"]["thinking"].append(results_1_1["thinking"])
        loop_results_stage_1["stage_1.subtask_1"]["answer"].append(results_1_1["answer"])
        logs.append(log1_1)

        cot_sc_instruction_1_2 = (
            "Sub-task 2: Derive the expression for ln(2) in terms of T_1 and T_2 from the Boltzmann relation. "
            "Input: user query and outputs (thinking and answer) from stage_0.subtask_1 and stage_1.subtask_1 (all iterations)."
        )
        final_decision_instruction_1_2 = (
            "Sub-task 2: Synthesize and choose the most consistent derived expression for ln(2) in terms of T_1 and T_2."
        )
        cot_sc_desc_1_2 = {
            "instruction": cot_sc_instruction_1_2,
            "final_decision_instruction": final_decision_instruction_1_2,
            "input": [taskInfo, results["stage_0.subtask_1"]["thinking"], results["stage_0.subtask_1"]["answer"]] + loop_results_stage_1["stage_1.subtask_1"]["thinking"] + loop_results_stage_1["stage_1.subtask_1"]["answer"] + loop_results_stage_1["stage_1.subtask_2"]["thinking"] + loop_results_stage_1["stage_1.subtask_2"]["answer"],
            "temperature": 0.5,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
        }
        results_1_2, log1_2 = await self.sc_cot(
            subtask_id="stage_1.subtask_2",
            cot_agent_desc=cot_sc_desc_1_2,
            n_repeat=self.max_sc if hasattr(self, 'max_sc') else 3
        )
        loop_results_stage_1["stage_1.subtask_2"]["thinking"].append(results_1_2["thinking"])
        loop_results_stage_1["stage_1.subtask_2"]["answer"].append(results_1_2["answer"])
        logs.append(log1_2)

    results["stage_1.subtask_1"] = {
        "thinking": loop_results_stage_1["stage_1.subtask_1"]["thinking"][-1],
        "answer": loop_results_stage_1["stage_1.subtask_1"]["answer"][-1]
    }
    results["stage_1.subtask_2"] = {
        "thinking": loop_results_stage_1["stage_1.subtask_2"]["thinking"][-1],
        "answer": loop_results_stage_1["stage_1.subtask_2"]["answer"][-1]
    }

    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Compare the derived ln(2) expression with the candidate equations and transform them algebraically to check consistency. "
        "Input: user query and outputs (thinking and answer) from stage_1.subtask_2."
    )
    critic_instruction_2_1 = (
        "Please review and provide the limitations of provided solutions comparing derived ln(2) expression and candidate equations."
    )
    cot_reflect_desc_2_1 = {
        "instruction": cot_reflect_instruction_2_1,
        "critic_instruction": critic_instruction_2_1,
        "input": [taskInfo, results["stage_1.subtask_2"]["thinking"], results["stage_1.subtask_2"]["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"]
    }
    results["stage_2.subtask_1"], log2_1 = await self.reflexion(
        subtask_id="stage_2.subtask_1",
        reflect_desc=cot_reflect_desc_2_1,
        n_repeat=getattr(self, 'max_round', 2)
    )
    logs.append(log2_1)

    debate_instruction_3_1 = (
        "Sub-task 1: Simplify and consolidate the transformed expressions and select the candidate equation that correctly models the temperature dependence. "
        "Input: user query and outputs (thinking and answer) from stage_1.subtask_2."
    )
    final_decision_instruction_3_1 = (
        "Sub-task 1: Select the best candidate equation for the effective temperatures T_1 and T_2 based on the analysis."
    )
    debate_desc_3_1 = {
        "instruction": debate_instruction_3_1,
        "final_decision_instruction": final_decision_instruction_3_1,
        "input": [taskInfo, results["stage_1.subtask_2"]["thinking"], results["stage_1.subtask_2"]["answer"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results["stage_3.subtask_1"], log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc_3_1,
        n_repeat=getattr(self, 'max_round', 2)
    )
    logs.append(log3_1)

    review_instruction_4_1 = (
        "Sub-task 1: Evaluate the selected candidate equation for correctness and consistency with physical principles and problem constraints. "
        "Input: user query and outputs (thinking and answer) from stage_2.subtask_1 and stage_3.subtask_1."
    )
    review_desc_4_1 = {
        "instruction": review_instruction_4_1,
        "input": [taskInfo, results["stage_2.subtask_1"]["thinking"], results["stage_2.subtask_1"]["answer"], results["stage_3.subtask_1"]["thinking"], results["stage_3.subtask_1"]["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_3.subtask_1", "answer of stage_3.subtask_1"]
    }
    results["stage_4.subtask_1"], log4_1 = await self.review(
        subtask_id="stage_4.subtask_1",
        review_desc=review_desc_4_1
    )
    logs.append(log4_1)

    final_answer = await self.make_final_answer(results["stage_4.subtask_1"]["thinking"], results["stage_4.subtask_1"]["answer"])
    return final_answer, logs
