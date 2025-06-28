async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify neutrino energy spectra bands and pp-III branch role, and time delay

    # Sub-task 1: Identify and characterize neutrino energy spectra bands 700-800 keV and 800-900 keV, and solar neutrino branches contributing
    cot_instruction_1 = (
        "Sub-task 1: Identify and characterize the neutrino energy spectra bands relevant to the problem: "
        "band 1 (700-800 keV) and band 2 (800-900 keV), including which solar neutrino production branches contribute to each band under normal conditions."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying neutrino energy bands, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 3: Determine time delay for neutrinos from solar core to Earth (~8.5 minutes)
    cot_instruction_3 = (
        "Sub-task 3: Determine the time delay for neutrinos produced in the solar core to reach Earth (~8.5 minutes), "
        "confirming that neutrinos observed now correspond to solar core conditions 8.5 minutes ago, and thus the hypothetical stopping of the pp-III branch 8.5 minutes ago would affect the current neutrino flux."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, determining neutrino travel time, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 2: Understand role and neutrino energy contribution of pp-III branch, which band(s) it affects and typical flux contribution
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, understand the role and neutrino energy contribution of the pp-III branch in the solar neutrino spectrum, "
        "specifically which energy band(s) it predominantly affects and the typical flux contribution it provides."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing pp-III branch role, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose most common answer for subtask 2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: Calculate flux changes and ratio after pp-III branch stops

    # Sub-task 4: Calculate expected change in neutrino flux in each band due to sudden stopping of pp-III branch 8.5 minutes ago
    cot_reflect_instruction_4 = (
        "Sub-task 4: Calculate the expected change in neutrino flux in each energy band (700-800 keV and 800-900 keV) at Earth due to the sudden stopping of the pp-III branch 8.5 minutes ago, "
        "assuming all other branches remain unchanged and ignoring neutrino flavor changes."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, calculating flux changes, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Critically evaluate the flux change calculation for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining flux changes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compute approximate ratio of flux (band 1) to flux (band 2) after pp-III branch stops
    debate_instruction_5 = (
        "Sub-task 5: Based on Sub-task 4 output, compute the approximate ratio of flux (band 1) to flux (band 2) after the pp-III branch stops, "
        "using the flux values derived in Sub-task 4."
    )
    debate_roles = ["Proponent", "Opponent"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
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
            agents.append(f"Debate agent {agent.id}, round {r}, computing flux ratio, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                    "Sub-task 5: Make final decision on the approximate flux ratio after pp-III branch stops.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final flux ratio, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare computed flux ratio to provided multiple-choice options and select closest
    cot_instruction_6 = (
        "Sub-task 6: Compare the computed flux ratio to the provided multiple-choice options (0.1, 10, 1, 0.01) "
        "and select the closest approximate value."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, selecting closest flux ratio option, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
