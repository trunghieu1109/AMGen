async def forward_0(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Sub-task 1: Identify and calculate Ayas walking speed (s) for 4 hours walk
    cot_instruction_1 = "Sub-task 1: Identify and calculate Ayas walking speed (s) when the walk takes 4 hours, including time spent in the coffee shop."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying and calculating walking speed, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Determine time spent at the coffee shop
    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, determine the time she spends at the coffee shop based on both conditions."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}

    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining time spent at the coffee shop, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2

    answer2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2]
    answer2 = answermapping_2[answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Calculate walking speed when walking at s+2 km/h
    cot_reflect_instruction = "Sub-task 3: Based on outputs from Sub-task 1 and Sub-task 2, calculate Ayas walking speed when she walks at s+2 km/h and determine the walk time excluding coffee shop time."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round

    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, calculating walking speed at s+2 km/h, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback3, correct3 = await critic_agent_3([taskInfo, thinking3, answer3], "please review the speed calculation at s+2 km/h and provide feedback.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback on speed s+2 km/h, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining walking speed at s+2 km/h, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Solve for time taken for the walk by creating equation
    debate_instruction_4 = "Sub-task 4: Use results from all stages to solve for walk time by creating an equation based on both scenarios and solving the system of equations."
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round

    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for agent in debate_agents_4:
            input_infos_4 = [taskInfo, thinking3, answer3]
            if r > 0:
                input_infos_4.extend(all_thinking4[r-1] + all_answer4[r-1])
            thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, creating and solving equations, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on time taken for the walk system of equations.", is_sub_task=True)
    agents.append(f"Final Decision agent, making decision on equations, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Validate correctness of s and t across scenarios
    cot_reflect_validate_instruction = "Sub-task 5: Validate correctness of s and t by ensuring consistency across different walking speeds."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round

    cot_inputs_5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_validate_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, validating consistency, thinking: {thinking5.content}; answer: {answer5.content}")

    for i in range(N_max_5):
        feedback5, correct5 = await critic_agent_5([taskInfo, thinking5, answer5], "please review the validation of consistency across walking speeds.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback on consistency validation, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_validate_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining consistency validation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Calculate time at speed s+1/2 km/h including coffee shop time
    cot_final_instruction = "Sub-task 6: Calculate new time taken for Ayas walk when she walks at s+1/2 km/h and include time spent at the coffee shop."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_final_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating time at speed s+1/2 km/h, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer