async def forward_7(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_bands = []

    cot_sc_instruction = (
        "Sub-task 1: Generate candidate Lyric Street Records-affiliated bands that could have covered 'If You Ever Get Lonely'. "
        "Explicitly verify band affiliations and cover versions from reliable sources, clearly distinguish original performers from cover artists, and provide references or evidence for each candidate."
    )

    for i in range(self.max_sc):
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.sc_cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results['cot_agent'][0].id}, iteration {i+1}, generating candidate bands with verified facts, thinking: {results['list_thinking'][0]}; answer: {results['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results['list_thinking'][0]}; answer - {results['list_answer'][0]}")
        logs.append(results['subtask_desc'])
        candidate_bands.append(results['list_answer'][0])

    cot_reflect_instruction = (
        "Sub-task 2: Evaluate the candidate bands generated in Sub-task 1 by assessing the quality, confidence, and factual correctness of each answer. "
        "Flag any conflicting answers and provide detailed reasoning or evidence for the final consolidated answer identifying the correct Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'."
    )
    critic_instruction = (
        "Please review the evaluation of candidate bands, identify any inconsistencies or errors, and provide feedback for improvement."
    )
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo] + candidate_bands,
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"] + [f"candidate band {i+1}" for i in range(len(candidate_bands))]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }

    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )

    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, evaluating candidate bands, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining evaluation, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction3 = (
        "Sub-task 3: Review the consolidated and evaluated answer to ensure it accurately and completely identifies the correct Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'. "
        "Provide any additional insights or corrections if necessary."
    )
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }

    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )

    agents.append(f"CoT agent {results3['cot_agent'].id}, reviewing consolidated answer, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    programmer_instruction4 = (
        "Sub-task 4: Generate Python runnable code that documents and validates the final answer identifying the Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'. "
        "The code should include references or evidence used in the reasoning process."
    )
    programmer_desc4 = {
        'instruction': programmer_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'entry_point': "validate_and_document_cover_band"
    }

    results4 = await self.programmer(
        subtask_id="subtask_4",
        programmer_desc=programmer_desc4
    )

    agents.append(f"Programmer Agent {results4['programmer_agent'].id}, generating validation code, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}, executing results: {results4['exec_result']}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}; output - {results4['exec_result']}")
    logs.append(results4['subtask_desc'])

    cot_instruction5 = (
        "Sub-task 5: Final reasoning to confirm and justify the validated answer identifying the Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'. "
        "Cite sources or evidence to strengthen the final confirmation."
    )
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }

    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )

    agents.append(f"CoT agent {results5['cot_agent'].id}, final confirmation, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
