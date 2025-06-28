async def forward_31(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify nucleus properties and mass
    # Sub-task 1: Identify Z, N, A
    instruction1 = "Sub-task 1: Given X is a Li nucleus with 3 neutrons, identify atomic number Z, neutron number N, and mass number A."
    agent1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await agent1([taskInfo], instruction1, is_sub_task=True)
    agents.append(f"CoT agent {agent1.id}, identifying Z, N, A, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Retrieve rest mass in atomic mass units with Self-Consistency CoT
    instruction2 = "Sub-task 2: Retrieve the standard rest mass of the 6Li nucleus in atomic mass units (u) from nuclear data tables, based on A from Sub-task 1."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers2 = []
    thinking_map2 = {}
    answer_map2 = {}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, retrieving rest mass, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers2.append(answer2_i.content)
        thinking_map2[answer2_i.content] = thinking2_i
        answer_map2[answer2_i.content] = answer2_i
    most_common2 = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinking_map2[most_common2]
    answer2 = answer_map2[most_common2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Convert rest mass from u to MeV/c² and GeV/c² with Reflexion
    instruction3 = "Sub-task 3: Convert the rest mass of 6Li from atomic mass units (u) to energy units (MeV/c²) and then into GeV/c²."
    cot3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot3(cot_inputs3, instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot3.id}, converting mass units, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic3([taskInfo, thinking3, answer3], "Critically evaluate the unit conversion for correctness and precision.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic3.id}, feedback on conversion, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot3(cot_inputs3, instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot3.id}, refining unit conversion, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Identify relativistic formulas
    instruction4 = "Sub-task 4: Identify and write down the relativistic formulas: Lorentz factor gamma = 1/sqrt(1 - v^2/c^2) for v=0.96c, and kinetic energy E_k = (gamma - 1) m c^2."
    agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await agent4([taskInfo], instruction4, is_sub_task=True)
    agents.append(f"CoT agent {agent4.id}, identifying formulas, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Compute Lorentz factor and kinetic energy, then select choice
    # Sub-task 5: Compute gamma
    instruction5 = "Sub-task 5: Using v=0.96c and formula from Sub-task 4, compute the Lorentz factor gamma with sufficient precision."
    agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await agent5([taskInfo, thinking4, answer4], instruction5, is_sub_task=True)
    agents.append(f"CoT agent {agent5.id}, computing Lorentz factor, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Calculate kinetic energy with Debate
    instruction6 = "Sub-task 6: Combine gamma from Sub-task 5 and rest mass m (in GeV/c²) from Sub-task 3 into E_k = (gamma - 1) * m c² and calculate kinetic energy in GeV with precision 1e-4."
    roles = ["Proposer", "Challenger"]
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in roles]
    all_thinking6 = []
    all_answer6 = []
    for r in range(self.max_round):
        round_thinking = []
        round_answer = []
        for agent in debate_agents6:
            if r == 0:
                inputs6 = [taskInfo, thinking3, answer3, thinking5, answer5]
            else:
                inputs6 = [taskInfo, thinking3, answer3, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
            thinking6_i, answer6_i = await agent(inputs6, instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing E_k, thinking: {thinking6_i.content}; answer: {answer6_i.content}")
            round_thinking.append(thinking6_i)
            round_answer.append(answer6_i)
        all_thinking6.append(round_thinking)
        all_answer6.append(round_answer)
    final_decision6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on calculated kinetic energy.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision6.id}, finalizing E_k, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Select closest choice
    instruction7 = "Sub-task 7: Compare computed kinetic energy from Sub-task 6 against choices: 18.475, 20.132, 23.069, 21.419 GeV and select the closest within 1e-4."
    agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await agent7([taskInfo, thinking6, answer6], instruction7, is_sub_task=True)
    agents.append(f"CoT agent {agent7.id}, selecting closest choice, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer