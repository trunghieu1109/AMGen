async def forward_55(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify initial, intermediate, and final states and verify allowed dipole transitions
    cot_instruction_1 = (
        "Sub-task 1: Identify the initial, intermediate, and final states involved in the two-step dipole transition "
        "from |3,0,0> to |1,0,0> in the hydrogen atom using standard notation |n,l,m>, and verify allowed dipole transitions "
        "based on selection rules (Δl = ±1, Δm = 0, ±1)."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying states and allowed transitions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1, Subtask 2: Determine all possible intermediate states |2,1,m> (m=-1,0,1) reachable from |3,0,0> and leading to |1,0,0>
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, determine all possible intermediate states |2,1,m> (m = -1, 0, 1) "
        "that can be reached from |3,0,0> by allowed dipole transitions and from which |1,0,0> can be reached by another allowed dipole transition."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining intermediate states, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Aggregate most consistent intermediate states
    counter_2 = Counter(possible_answers_2)
    most_common_2 = counter_2.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinkingmapping_2[most_common_2].content}; answer - {most_common_2}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2, Subtask 3: Calculate dipole transition probabilities for each step
    cot_reflect_instruction_3 = (
        "Sub-task 3: Calculate the dipole transition probabilities (or relative strengths) for each step of the two-step transition route: "
        "from |3,0,0> to each allowed intermediate state |2,1,m>, and from each intermediate state to |1,0,0>, using dipole selection rules and relevant quantum mechanical formulas."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinkingmapping_2[most_common_2], answermapping_2[most_common_2]]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, calculating transition probabilities, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Critically evaluate the dipole transition probability calculations for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining transition probabilities, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2, Subtask 4: Combine transition probabilities for two-step routes and express results in LaTeX
    cot_instruction_4 = (
        "Sub-task 4: Combine the transition probabilities of the two steps for each possible route to find the total probability of the two-step dipole transition "
        "from |3,0,0> to |1,0,0> via each intermediate state |2,1,m>, and express the results in LaTeX format."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, combining probabilities and formatting LaTeX, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 3, Subtask 5: Compare calculated routes and probabilities with given choices and identify correct choice
    debate_instruction_5 = (
        "Sub-task 5: Based on the output of Sub-task 4, compare the calculated transition routes and their corresponding probabilities with the given choices, "
        "and identify which choice correctly represents the transition route and probability in LaTeX notation."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing choices and identifying correct one, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct transition route and probability.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on correct choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
