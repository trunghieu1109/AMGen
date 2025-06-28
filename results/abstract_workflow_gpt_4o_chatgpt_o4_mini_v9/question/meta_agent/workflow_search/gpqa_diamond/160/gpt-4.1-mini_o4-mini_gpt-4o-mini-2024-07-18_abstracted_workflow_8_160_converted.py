async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    cot_instruction_1 = (
        "Sub-task 1: Understand and define the physical meaning of the mean free path (λ) of gas molecules "
        "in an ultra-high vacuum environment (< 10^-9 Torr) inside the sample compartment, considering the given parameters: volume, pressure, and temperature. "
        "This sets the baseline λ1 before electron beam initiation."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, understanding mean free path λ1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze how the presence of an electron beam at 1000 kV accelerating voltage affects the interaction between electrons and residual gas molecules, "
        "potentially altering the effective mean free path (λ2) related to electron scattering events, while temperature remains constant. "
        "Consider electron-gas molecule scattering physics and its impact on λ2."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing electron beam effect on λ2, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    counter_2 = Counter(possible_answers_2)
    most_common_answer_2 = counter_2.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 2 output: most consistent answer - {most_common_answer_2}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Sub-task 3: Quantitatively compare λ1 (mean free path of gas molecules without electron beam) and λ2 (mean free path related to electron scattering with gas molecules) "
        "by considering the physics of electron-gas molecule scattering, including cross sections and scattering probabilities, to determine the expected relationship between λ1 and λ2."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinkingmapping_2[most_common_answer_2], answermapping_2[most_common_answer_2]]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, comparing λ1 and λ2 quantitatively, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Please review the quantitative comparison of λ1 and λ2, check for physical correctness and completeness, and provide limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining comparison, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Based on the comparison and physical principles, debate and conclude which of the provided choices "
        "(λ2 = λ1, λ1 < λ2 < 1.22*λ1, λ2 < λ1, or λ2 >= 1.22*λ1) correctly describes the relationship between λ2 and λ1 in the given scenario. "
        "Consider all previous outputs and arguments."
    )
    debate_roles = ["Pro-λ2=λ1", "Pro-λ1<λ2<1.22λ1", "Pro-λ2<λ1", "Pro-λ2>=1.22λ1"]
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating correct λ2 relation, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1],
                                                    "Sub-task 4: Make final decision on which choice correctly describes the relationship between λ2 and λ1.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, making final conclusion on λ2 vs λ1, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
