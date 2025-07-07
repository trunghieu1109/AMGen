async def forward_7(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_bands = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop
    for i in range(3):
        cot_instruction = f"Subtask {i+1}: Systematically generate a candidate band affiliated with Lyric Street Records that covered 'If You Ever Get Lonely' by applying structured reasoning"
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.answer_generate(
            subtask_id=f"subtask_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, generating candidate band #{i+1}, thinking: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Subtask {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        candidate_bands.append(results['answer'].content)
    # Control Flow 2: end loop
    aggregate_instruction = "Subtask 4: Integrate multiple candidate bands by evaluating their consistency and synthesizing them into a single coherent output identifying the band that covered 'If You Ever Get Lonely'"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_bands,
        'temperature': 0.0,
        'context': ["user query", "candidate bands generated"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate bands, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    review_instruction = "Subtask 5: Evaluate the consolidated output against established criteria to confirm its accuracy, completeness, validity, and correctness"
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing consolidated output, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs