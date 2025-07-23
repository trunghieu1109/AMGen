async def forward_175(self, taskInfo):
    logs = []
    loop_results = {"stage_0.subtask_1": {"thinking": [], "answer": []},
                    "stage_0.subtask_2": {"thinking": [], "answer": []},
                    "stage_0.subtask_3": {"thinking": [], "answer": []},
                    "stage_0.subtask_4": {"thinking": [], "answer": []},
                    "stage_0.subtask_5": {"thinking": [], "answer": []}}

    for iteration in range(2):
        cot_agent_desc_1 = {
            "instruction": "Sub-task 1: Normalize the given initial state vector and verify normalization. Input: taskInfo.",
            "input": [taskInfo],
            "temperature": 0.0,
            "context_desc": ["user query"]
        }
        results1, log1 = await self.cot(subtask_id="stage_0.subtask_1", cot_agent_desc=cot_agent_desc_1)
        loop_results["stage_0.subtask_1"]["thinking"].append(results1["thinking"])
        loop_results["stage_0.subtask_1"]["answer"].append(results1["answer"])
        logs.append(log1)

        cot_agent_desc_2 = {
            "instruction": "Sub-task 2: Find eigenvalues and eigenvectors of operator P, identify eigenvector(s) corresponding to eigenvalue 0. Input: taskInfo, all previous thinking and answers from stage_0.subtask_1.",
            "input": [taskInfo] + loop_results["stage_0.subtask_1"]["thinking"] + loop_results["stage_0.subtask_1"]["answer"],
            "temperature": 0.0,
            "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
        }
        results2, log2 = await self.cot(subtask_id="stage_0.subtask_2", cot_agent_desc=cot_agent_desc_2)
        loop_results["stage_0.subtask_2"]["thinking"].append(results2["thinking"])
        loop_results["stage_0.subtask_2"]["answer"].append(results2["answer"])
        logs.append(log2)

        cot_agent_desc_3 = {
            "instruction": "Sub-task 3: Project the normalized initial state onto the eigenspace of P with eigenvalue 0 to get the post-measurement state after measuring P. Input: all previous thinking and answers from stage_0.subtask_2.",
            "input": loop_results["stage_0.subtask_2"]["thinking"] + loop_results["stage_0.subtask_2"]["answer"],
            "temperature": 0.0,
            "context_desc": ["thinking of stage_0.subtask_2", "answer of stage_0.subtask_2"]
        }
        results3, log3 = await self.cot(subtask_id="stage_0.subtask_3", cot_agent_desc=cot_agent_desc_3)
        loop_results["stage_0.subtask_3"]["thinking"].append(results3["thinking"])
        loop_results["stage_0.subtask_3"]["answer"].append(results3["answer"])
        logs.append(log3)

        cot_agent_desc_4 = {
            "instruction": "Sub-task 4: Find eigenvalues and eigenvectors of operator Q, identify eigenvector corresponding to eigenvalue -1. Input: taskInfo.",
            "input": [taskInfo],
            "temperature": 0.0,
            "context_desc": ["user query"]
        }
        results4, log4 = await self.cot(subtask_id="stage_0.subtask_4", cot_agent_desc=cot_agent_desc_4)
        loop_results["stage_0.subtask_4"]["thinking"].append(results4["thinking"])
        loop_results["stage_0.subtask_4"]["answer"].append(results4["answer"])
        logs.append(log4)

        cot_agent_desc_5 = {
            "instruction": "Sub-task 5: Calculate the probability of measuring eigenvalue 0 for P and then eigenvalue -1 for Q sequentially using the post-measurement state. Input: all previous thinking and answers from stage_0.subtask_3 and stage_0.subtask_4.",
            "input": loop_results["stage_0.subtask_3"]["thinking"] + loop_results["stage_0.subtask_3"]["answer"] + loop_results["stage_0.subtask_4"]["thinking"] + loop_results["stage_0.subtask_4"]["answer"],
            "temperature": 0.0,
            "context_desc": ["thinking of stage_0.subtask_3", "answer of stage_0.subtask_3", "thinking of stage_0.subtask_4", "answer of stage_0.subtask_4"]
        }
        results5, log5 = await self.cot(subtask_id="stage_0.subtask_5", cot_agent_desc=cot_agent_desc_5)
        loop_results["stage_0.subtask_5"]["thinking"].append(results5["thinking"])
        loop_results["stage_0.subtask_5"]["answer"].append(results5["answer"])
        logs.append(log5)

    cot_sc_desc_1 = {
        "instruction": "Sub-task 1: Simplify and consolidate the probability expression obtained, ensuring correct normalization and interpretation. Input: all thinking and answers from stage_0.subtask_5.",
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent simplified probability expression.",
        "input": [taskInfo] + loop_results["stage_0.subtask_5"]["thinking"] + loop_results["stage_0.subtask_5"]["answer"],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results_sc, log_sc = await self.sc_cot(subtask_id="stage_1.subtask_1", cot_agent_desc=cot_sc_desc_1, n_repeat=self.max_sc)
    logs.append(log_sc)

    cot_agent_desc_2 = {
        "instruction": "Sub-task 1: Apply any necessary algebraic or numerical transformations to finalize the probability value in simplest form. Input: thinking and answer from stage_1.subtask_1.",
        "input": [taskInfo, results_sc["thinking"], results_sc["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.cot(subtask_id="stage_2.subtask_1", cot_agent_desc=cot_agent_desc_2)
    logs.append(log2)

    review_desc = {
        "instruction": "Sub-task 1: Validate the final probability result against quantum measurement postulates and consistency with given multiple-choice options. Input: thinking and answer from stage_1.subtask_1 and stage_2.subtask_1.",
        "input": [taskInfo, results_sc["thinking"], results_sc["answer"], results2["thinking"], results2["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"]
    }
    results_review, log_review = await self.review(subtask_id="stage_3.subtask_1", review_desc=review_desc)
    logs.append(log_review)

    final_answer = await self.make_final_answer(results2["thinking"], results2["answer"])
    return final_answer, logs
