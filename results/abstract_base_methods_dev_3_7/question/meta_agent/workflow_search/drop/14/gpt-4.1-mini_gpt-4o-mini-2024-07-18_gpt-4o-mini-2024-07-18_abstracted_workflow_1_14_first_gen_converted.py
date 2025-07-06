async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Extract all racial groups and their corresponding population percentages from the passage provided in taskInfo."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting racial groups and percentages, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    racial_groups = results1['answer'].content
    filtered_groups = []
    for idx, group_info in enumerate(racial_groups):
        cot_instruction2 = f"Subtask 2: For the racial group '{group_info}', check if its population percentage is between 0.5% and 2.5%."
        cot_agent_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, group_info],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{idx+1}",
            cot_agent_desc=cot_agent_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, checking percentage range for {group_info}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2 output for {group_info}: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        if 'yes' in results2['answer'].content.lower():
            cot_instruction3 = f"Subtask 3: Add the racial group '{group_info}' to the list of groups meeting the 0.5% to 2.5% population criterion."
            cot_agent_desc3 = {
                'instruction': cot_instruction3,
                'input': [taskInfo, group_info],
                'temperature': 0.0,
                'context': ["user query"]
            }
            results3 = await self.cot(
                subtask_id=f"subtask_3_{idx+1}",
                cot_agent_desc=cot_agent_desc3
            )
            agents.append(f"CoT agent {results3['cot_agent'].id}, adding group {group_info}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
            sub_tasks.append(f"Subtask 3 output for {group_info}: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
            logs.append(results3['subtask_desc'])
            filtered_groups.append(group_info)
    cot_instruction4 = "Subtask 4: Compile and produce the final list of racial groups that individually make up between 0.5% and 2.5% of the population."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, filtered_groups],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, compiling final list, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs