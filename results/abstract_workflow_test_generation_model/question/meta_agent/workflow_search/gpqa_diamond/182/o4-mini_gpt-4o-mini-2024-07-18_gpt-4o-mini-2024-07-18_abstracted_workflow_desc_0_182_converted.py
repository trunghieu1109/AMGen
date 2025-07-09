async def forward_182(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task subtask_0_1: Analyze the structure of 2-formyl-5-vinylcyclohex-3-enecarboxylic acid to determine its carbon skeleton, ring system, positions of double bonds, and substituent functional groups."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    subtask_desc = {"subtask_id":"subtask_0_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing structure, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task subtask_0_1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc['response']={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task subtask_1_1: Identify all chemical transformations that occur when the starting molecule is reacted with red phosphorus and excess HI."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc = {"subtask_id":"subtask_1_1","instruction":cot_sc_instruction,"context":["user query","thinking of subtask_0_1","answer of subtask_0_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, identifying transformations, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_map[answer2.content] = thinking2
        answer_map[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_map[answer2_content]
    answer2 = answer_map[answer2_content]
    sub_tasks.append(f"Sub-task subtask_1_1 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc['response']={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task subtask_1_2: Determine how each transformation alters the counts of carbon and hydrogen atoms and total degree of unsaturation in the molecule."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc = {"subtask_id":"subtask_1_2","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask_0_1","answer of subtask_0_1","thinking of subtask_1_1","answer of subtask_1_1"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, determining alterations, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3],"Review the atom count and IHD alterations and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs += [thinking3, answer3, feedback]
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining alterations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task subtask_1_2 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc['response']={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task subtask_2_1: Compile the molecular formula of the final product by applying net changes to the starting structure."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc = {"subtask_id":"subtask_2_1","instruction":cot_sc_instruction,"context":["user query","thinking of subtask_1_2","answer of subtask_1_2"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking4, answer4 = await agent([taskInfo, thinking3, answer3], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, compiling formula, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers.append(answer4.content)
        thinking_map[answer4.content] = thinking4
        answer_map[answer4.content] = answer4
    answer4_content = Counter(possible_answers).most_common(1)[0][0]
    thinking4 = thinking_map[answer4_content]
    answer4 = answer_map[answer4_content]
    sub_tasks.append(f"Sub-task subtask_2_1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc['response']={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task subtask_3_1: Calculate the index of hydrogen deficiency (IHD) for the product using its molecular formula (IHD = C – H/2 + 1)."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_map = {}
    answer_map = {}
    subtask_desc = {"subtask_id":"subtask_3_1","instruction":cot_sc_instruction,"context":["user query","thinking of subtask_2_1","answer of subtask_2_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking5, answer5 = await agent([taskInfo, thinking4, answer4], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, calculating IHD, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers.append(answer5.content)
        thinking_map[answer5.content] = thinking5
        answer_map[answer5.content] = answer5
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinking_map[answer5_content]
    answer5 = answer_map[answer5_content]
    sub_tasks.append(f"Sub-task subtask_3_1 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc['response']={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task subtask_3_2: Compare the calculated IHD value to choices (1,3,0,5) and select the corresponding letter A,B,C,or D."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking5, answer5]
    subtask_desc = {"subtask_id":"subtask_3_2","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask_3_1","answer of subtask_3_1"],"agent_collaboration":"Reflexion"}
    thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, selecting answer choice, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking6, answer6],"Review the choice mapping and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs += [thinking6, answer6, feedback]
        thinking6, answer6 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining choice selection, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task subtask_3_2 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc['response']={"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs