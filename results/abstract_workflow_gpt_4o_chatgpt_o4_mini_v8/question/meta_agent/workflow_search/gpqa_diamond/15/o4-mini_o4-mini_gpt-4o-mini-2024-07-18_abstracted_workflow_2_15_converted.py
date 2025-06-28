async def forward_15(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    cot_instruction = "Sub-task 1: Extract and list the seven compound names from the query."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting compound names, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Determine all stereogenic elements (chiral centers and E/Z double bonds) for each compound."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, identifying stereogenic elements, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinkingmapping2[answer2_i.content] = thinking2_i
        answermapping2[answer2_i.content] = answer2_i
    majority_answer2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[majority_answer2]
    answer2 = answermapping2[majority_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])
    cot_reflect3 = "Sub-task 3: Classify each compound by count and type of stereogenic elements."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, classifying stereogenic elements, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic3([taskInfo, thinking3, answer3], "Review the classification and identify any mistakes.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic3.id}, feedback on classification, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining classification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Assess actual chirality by checking symmetry, meso forms, and racemic possibilities."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, assessing chirality, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])
    cot_instruction5 = "Sub-task 5: Tally the total number of compounds predicted to be optically active."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, tallying optically active compounds, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])
    debate_instruction6 = "Sub-task 6: Compare the computed total against choices (2,3,4,5) and select the correct answer."
    debate_agents6 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking6 = [[] for _ in range(self.max_round)]
    all_answer6 = [[] for _ in range(self.max_round)]
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents6):
            if r==0:
                thinking6_i, answer6_i = await agent([taskInfo, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                inputs6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6_i, answer6_i = await agent(inputs6, debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final choice, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            all_thinking6[r].append(thinking6_i)
            all_answer6[r].append(answer6_i)
    final_decision6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the number of optically active compounds.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer