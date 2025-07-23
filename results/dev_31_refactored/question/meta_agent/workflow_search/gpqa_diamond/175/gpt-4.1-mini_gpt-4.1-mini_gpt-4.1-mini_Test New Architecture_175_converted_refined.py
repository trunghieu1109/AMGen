async def forward_175(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_0.subtask_3": {"thinking": [], "answer": []},
                    "stage_0.subtask_4": {"thinking": [], "answer": []},
                    "stage_0.subtask_5": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Normalize the given initial state vector and verify normalization explicitly, "
            "noting that incorrect normalization was a root cause of previous errors. "
            "Input content: taskInfo (user query with initial state vector)."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id="stage_0.subtask_1",
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results["stage_0.subtask_1"]["thinking"].append(results_0_1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results_0_1["answer"])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Find eigenvalues and eigenvectors of operator P, identify eigenvector(s) corresponding to eigenvalue 0, "
            "and carefully verify correctness of eigen-decomposition to avoid propagation of errors. "
            "Input content: taskInfo, thinking and answer from stage_0.subtask_1."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id="stage_0.subtask_2",
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results["stage_0.subtask_2"]["thinking"].append(results_0_2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results_0_2["answer"])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Project the normalized initial state onto the eigenspace of P with eigenvalue 0 to get the post-measurement state after measuring P, "
            "ensuring correct projection and normalization of the post-measurement state to prevent errors in subsequent probability calculations. "
            "Input content: thinking and answer from stage_0.subtask_2 and stage_0.subtask_1."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo] + loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id="stage_0.subtask_3",
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results["stage_0.subtask_3"]["thinking"].append(results_0_3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results_0_3["answer"])
        logs.append(log_0_3)

        cot_instruction_0_4 = (
            "Sub-task 4: Find eigenvalues and eigenvectors of operator Q, identify eigenvector corresponding to eigenvalue -1, "
            "and verify eigen-decomposition accuracy to avoid mistakes in probability evaluation. "
            "Input content: thinking and answer from stage_0.subtask_1."
        )
        cot_agent_desc_0_4 = {
            "instruction": cot_instruction_0_4,
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id="stage_0.subtask_4",
            cot_agent_desc=cot_agent_desc_0_4
        )
        loop_results["stage_0.subtask_4"]["thinking"].append(results_0_4["thinking"])
        loop_results["stage_0.subtask_4"]["answer"].append(results_0_4["answer"])
        logs.append(log_0_4)

        cot_instruction_0_5 = (
            "Sub-task 5: Calculate the joint probability of measuring eigenvalue 0 for P and then eigenvalue -1 for Q sequentially using the post-measurement state, "
            "explicitly applying quantum measurement postulates and ensuring all states are normalized; note that previous errors stemmed from insufficient scrutiny of this calculation. "
            "Input content: thinking and answer from stage_0.subtask_3 and stage_0.subtask_4."
        )
        cot_agent_desc_0_5 = {
            "instruction": cot_instruction_0_5,
            "input": [taskInfo] + loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"] + loop_results["stage_0.subtask_4"]["thinking"] + loop_results["stage_0.subtask_4"]["answer"],
            "temperature": 0.0,
            "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results_0_5, log_0_5 = await self.cot(
            subtask_id="stage_0.subtask_5",
            cot_agent_desc=cot_agent_desc_0_5
        )
        loop_results["stage_0.subtask_5"]["thinking"].append(results_0_5["thinking"])
        loop_results["stage_0.subtask_5"]["answer"].append(results_0_5["answer"])
        logs.append(log_0_5)

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Simplify and consolidate the probability expression obtained, ensuring correct normalization and interpretation; "
        "explicitly check for consistency with quantum mechanical principles and previous steps to avoid repeating past mistakes. "
        "Input content: thinking and answer from stage_0.subtask_5 (all iterations)."
    )
    final_decision_instruction_1_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the probability calculation, "
        "based on all previous outputs from stage_0.subtask_5."
    )
    cot_sc_desc_1_1 = {
        "instruction": cot_sc_instruction_1_1,
        "final_decision_instruction": final_decision_instruction_1_1,
        "input": [taskInfo] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_0.subtask_5", "answer of stage_0.subtask_5"]
    }
    results_1_1, log_1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc_1_1,
        n_repeat=self.max_sc
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Map the simplified numeric probability to the closest multiple-choice option (1/2, 1/6, 1/3, 2/3) and prepare for validation; "
        "this step addresses the previous lack of answer mapping and comparison to the official key. "
        "Input content: thinking and answer from stage_1.subtask_1."
    )
    cot_agent_desc_1_2 = {
        "instruction": cot_instruction_1_2,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_instruction_2_1 = (
        "Sub-task 1: Apply any necessary algebraic or numerical transformations to finalize the probability value in simplest form, "
        "ensuring the result is in a format directly comparable to the multiple-choice options; "
        "this step supports clarity and correctness in final answer presentation. "
        "Input content: thinking and answer from stage_1.subtask_1."
    )
    cot_agent_desc_2_1 = {
        "instruction": cot_instruction_2_1,
        "input": [taskInfo, results_1_1["thinking"], results_1_1["answer"]],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results_2_1, log_2_1 = await self.cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_agent_desc_2_1
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Sub-task 1: Validate the final probability result against quantum measurement postulates and consistency with the given multiple-choice options; "
        "explicitly compare the mapped answer from stage_1.subtask_2 with the professor’s official key or expected answer. "
        "If a discrepancy is found, trigger a reanalysis by revisiting normalization, eigen-decomposition, and projection subtasks to identify and correct errors, "
        "thus preventing blind acceptance of incorrect results as happened previously. "
        "Input content: thinking and answer from stage_1.subtask_2, stage_2.subtask_1, stage_0.subtask_1, stage_0.subtask_2, stage_0.subtask_3."
    )
    review_desc_3_1 = {
        "instruction": review_instruction_3_1,
        "input": [taskInfo, results_1_2["thinking"], results_1_2["answer"], results_2_1["thinking"], results_2_1["answer"]] + 
                 loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"] + 
                 loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"] + 
                 loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_0.subtask_2", "answer of stage_0.subtask_2", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"]
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id="stage_3.subtask_1",
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1["thinking"], results_3_1["answer"])
    return final_answer, logs
