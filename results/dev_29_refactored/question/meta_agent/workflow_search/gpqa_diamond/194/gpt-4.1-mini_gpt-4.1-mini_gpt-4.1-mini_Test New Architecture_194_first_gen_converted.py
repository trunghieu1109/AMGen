async def forward_194(self, taskInfo):
    logs = []
    stage0_results = {}

    for iteration in range(3):
        iteration_results = {}

        cot_instruction_0_0 = (
            "Sub-task 0: Extract and summarize all given parameters and assumptions from the query to establish a clear problem statement."
        )
        cot_agent_desc_0_0 = {
            "instruction": cot_instruction_0_0,
            "input": [taskInfo],
            "temperature": 0.0,
            "context": ["user query"]
        }
        results_0_0, log_0_0 = await self.cot(
            subtask_id=f"stage_0_subtask_0_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_0
        )
        logs.append(log_0_0)
        iteration_results['subtask_0'] = results_0_0

        cot_instruction_0_1 = (
            "Sub-task 1: Derive the inclination of the orbit from the first planet's transit impact parameter and stellar radius."
        )
        cot_agent_desc_0_1 = {
            "instruction": cot_instruction_0_1,
            "input": [taskInfo, results_0_0['thinking'], results_0_0['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id=f"stage_0_subtask_1_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_1
        )
        logs.append(log_0_1)
        iteration_results['subtask_1'] = results_0_1

        cot_instruction_0_2 = (
            "Sub-task 2: Calculate the orbital radius of the first planet using its orbital period and estimated stellar mass (assumed or derived from stellar radius)."
        )
        cot_agent_desc_0_2 = {
            "instruction": cot_instruction_0_2,
            "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id=f"stage_0_subtask_2_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_2
        )
        logs.append(log_0_2)
        iteration_results['subtask_2'] = results_0_2

        cot_instruction_0_3 = (
            "Sub-task 3: Formulate the geometric conditions for the second planet to exhibit both transit and occultation events based on inclination, stellar radius, and planet radius."
        )
        cot_agent_desc_0_3 = {
            "instruction": cot_instruction_0_3,
            "input": [taskInfo, results_0_1['thinking'], results_0_1['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id=f"stage_0_subtask_3_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_3
        )
        logs.append(log_0_3)
        iteration_results['subtask_3'] = results_0_3

        cot_instruction_0_4 = (
            "Sub-task 4: Express the maximum orbital radius (and thus maximum orbital period) for the second planet that satisfies the geometric transit and occultation conditions."
        )
        cot_agent_desc_0_4 = {
            "instruction": cot_instruction_0_4,
            "input": [taskInfo, results_0_3['thinking'], results_0_3['answer'], results_0_2['thinking'], results_0_2['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2", "answer of subtask 2"]
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id=f"stage_0_subtask_4_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_4
        )
        logs.append(log_0_4)
        iteration_results['subtask_4'] = results_0_4

        cot_instruction_0_5 = (
            "Sub-task 5: Convert the maximum orbital radius to maximum orbital period using Kepler's third law and stellar mass assumptions."
        )
        cot_agent_desc_0_5 = {
            "instruction": cot_instruction_0_5,
            "input": [taskInfo, results_0_4['thinking'], results_0_4['answer']],
            "temperature": 0.0,
            "context": ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results_0_5, log_0_5 = await self.cot(
            subtask_id=f"stage_0_subtask_5_iter_{iteration}",
            cot_agent_desc=cot_agent_desc_0_5
        )
        logs.append(log_0_5)
        iteration_results['subtask_5'] = results_0_5

        aggregate_instruction_0_6 = (
            "Sub-task 6: Refine and consolidate all intermediate results and reasoning into a clear, coherent output that can be used for final candidate evaluation."
        )
        aggregate_desc_0_6 = {
            "instruction": aggregate_instruction_0_6,
            "input": [taskInfo, 
                      iteration_results['subtask_0']['thinking'], iteration_results['subtask_0']['answer'],
                      iteration_results['subtask_1']['thinking'], iteration_results['subtask_1']['answer'],
                      iteration_results['subtask_2']['thinking'], iteration_results['subtask_2']['answer'],
                      iteration_results['subtask_3']['thinking'], iteration_results['subtask_3']['answer'],
                      iteration_results['subtask_4']['thinking'], iteration_results['subtask_4']['answer'],
                      iteration_results['subtask_5']['thinking'], iteration_results['subtask_5']['answer']
                     ],
            "temperature": 0.0,
            "context": ["user query", "solutions generated from subtasks 0 to 5"]
        }
        results_0_6, log_0_6 = await self.aggregate(
            subtask_id=f"stage_0_subtask_6_iter_{iteration}",
            aggregate_desc=aggregate_desc_0_6
        )
        logs.append(log_0_6)
        iteration_results['subtask_6'] = results_0_6

        stage0_results[iteration] = iteration_results

    cot_agent_instruction_1_0 = (
        "Sub-task 0: Compare the calculated maximum orbital period with the given candidate choices (~7.5, ~33.5, ~37.5, ~12.5 days) to identify the best matching option."
    )
    cot_agent_desc_1_0 = {
        "instruction": cot_agent_instruction_1_0,
        "input": [taskInfo, 
                  stage0_results[2]['subtask_6']['thinking'], stage0_results[2]['subtask_6']['answer']
                 ],
        "temperature": 0.0,
        "context": ["user query", "thinking of stage_0 subtask_6 iteration 2", "answer of stage_0 subtask_6 iteration 2"]
    }
    results_1_0, log_1_0 = await self.answer_generate(
        subtask_id="stage_1_subtask_0",
        cot_agent_desc=cot_agent_desc_1_0
    )
    logs.append(log_1_0)

    final_answer = await self.make_final_answer(results_1_0['thinking'], results_1_0['answer'])

    return final_answer, logs
