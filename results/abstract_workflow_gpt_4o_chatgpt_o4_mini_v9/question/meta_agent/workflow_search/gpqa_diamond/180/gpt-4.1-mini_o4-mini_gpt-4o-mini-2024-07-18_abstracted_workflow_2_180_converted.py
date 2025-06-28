async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify key physical concepts (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Analyze the given query to identify and classify the key physical concepts involved, "
        "including solar neutrino production mechanisms, the specific pp-III branch, neutrino energy bands 700-800 keV and 800-900 keV, "
        "and the meaning of neutrino flux as number per cm^2 per second reaching Earth. Clarify the significance of the hypothetical stopping of the pp-III branch 8.5 minutes ago and ignoring flavor changes."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing key physical concepts, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 1: Extract and summarize relevant features of solar neutrino spectrum (Self-Consistency CoT)
    cot_sc_instruction_2 = (
        "Sub-task 2: Based on Sub-task 1 output, extract and summarize the relevant features of the solar neutrino spectrum, "
        "focusing on energy ranges 700-800 keV and 800-900 keV, and identify which neutrino production branches contribute to these bands, "
        "especially the contribution of the pp-III branch to the 800-900 keV band."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, extracting neutrino spectrum features, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Choose the most consistent answer (majority vote)
    from collections import Counter
    answer_counter_2 = Counter(possible_answers_2)
    best_answer_2 = answer_counter_2.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer_2]
    answer2 = answermapping_2[best_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Evaluate time delay and effect of stopping pp-III branch 8.5 minutes ago (CoT)
    cot_instruction_3 = (
        "Sub-task 3: Evaluate the time delay between neutrino production in the solar core and their arrival at Earth (~8.5 minutes) "
        "and understand how stopping the pp-III branch 8.5 minutes ago would affect the neutrino flux observed now on Earth, "
        "considering neutrinos travel at nearly the speed of light."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, evaluating time delay and flux effect, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Stage 2: Calculate expected change in neutrino flux in bands after pp-III branch stops (Reflexion)
    cot_reflect_instruction_4 = (
        "Sub-task 4: Using information about which neutrino energy bands are affected by the pp-III branch, "
        "calculate the expected change in neutrino flux in the 700-800 keV band (band 1) and the 800-900 keV band (band 2) after the pp-III branch stops, "
        "assuming all other branches remain unchanged."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking2, answer2, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, calculating flux changes, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "please review the flux change calculation for correctness and completeness.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining flux changes, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Compute approximate ratio Flux(band 1)/Flux(band 2) after pp-III stops and interpret (Debate)
    debate_instruction_5 = (
        "Sub-task 5: Based on Sub-task 4 output, compute the approximate ratio of fluxes Flux(band 1) / Flux(band 2) "
        "under the condition that the pp-III branch stopped 8.5 minutes ago, using flux values before and after stoppage, "
        "and interpret the result in context of the provided multiple-choice answers."
    )
    debate_roles = ["Proponent", "Skeptic"]
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
            agents.append(f"Debate agent {agent.id}, round {r}, computing flux ratio and interpretation, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1],
                                                     "Sub-task 5: Make final decision on the approximate ratio Flux(band 1) / Flux(band 2) and select the best matching multiple-choice answer.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final flux ratio answer, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
