async def forward_137(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize defining features
    cot_instruction_1 = (
        "Sub-task 1: Extract and characterize the defining features of the chemical reaction system, "
        "including initial conditions: presence of H+ ions, room temperature, initial pH of 1, and the exothermic reaction causing container heating. "
        "Analyze with context from taskInfo."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting defining features, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and isolate changes caused by accidental addition
    cot_sc_instruction_2 = (
        "Sub-task 2: Identify and isolate the changes caused by the accidental addition of an unknown substance, "
        "specifically the slowing down of the reaction rate, pH change from 1 to 4, and the exothermic nature after addition, "
        "based on outputs from Sub-task 1."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, isolating changes, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    # Majority vote for best answer
    answer2_counter = Counter(possible_answers_2)
    best_answer2 = answer2_counter.most_common(1)[0][0]
    thinking2 = thinkingmapping_2[best_answer2]
    answer2 = answermapping_2[best_answer2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Analyze effects of pH change, temperature, and other factors
    # Sub-task 3: Analyze pH change effect on H+ concentration and reaction rate
    cot_instruction_3 = (
        "Sub-task 3: Analyze how the pH change from 1 to 4 affects the concentration of H+ ions and how this influences the reaction rate, "
        "considering the reaction initially involved H+ ions, based on Sub-task 2 outputs."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing pH effect, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Assess impact of exothermic reaction (temperature increase) on reaction rate
    cot_instruction_4 = (
        "Sub-task 4: Assess the impact of the exothermic reaction causing container heating on the reaction rate, "
        "considering temperature generally increases reaction rates but the observed rate slowed down, based on Sub-task 2 outputs."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, assessing temperature impact, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Evaluate other factors such as pressure or volume changes
    cot_instruction_5 = (
        "Sub-task 5: Evaluate other possible factors such as changes in pressure or volume due to accidental addition, "
        "and their potential effects on the reaction rate, based on Sub-task 2 outputs."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, evaluating pressure/volume impact, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Classify and determine most plausible reason
    # Sub-task 6: Classify and compare possible reasons for reaction rate change
    cot_sc_instruction_6 = (
        "Sub-task 6: Classify and compare possible reasons for the change in reaction rate (increased pressure, volume, temperature, pH) "
        "based on analysis from Sub-tasks 3, 4, and 5 and observed data (pH change, rate decrease, exothermic heat)."
    )
    N6 = self.max_sc
    cot_agents_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers_6 = []
    thinkingmapping_6 = {}
    answermapping_6 = {}
    for i in range(N6):
        thinking6, answer6 = await cot_agents_6[i](
            [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5],
            cot_sc_instruction_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6[i].id}, classifying reasons, thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers_6.append(answer6.content)
        thinkingmapping_6[answer6.content] = thinking6
        answermapping_6[answer6.content] = answer6
    answer6_counter = Counter(possible_answers_6)
    best_answer6 = answer6_counter.most_common(1)[0][0]
    thinking6 = thinkingmapping_6[best_answer6]
    answer6 = answermapping_6[best_answer6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Determine most plausible reason for slowing down of reaction rate
    debate_instruction_7 = (
        "Sub-task 7: Based on Sub-task 6 output, debate and determine the most plausible reason for the slowing down of the reaction rate after accidental addition, "
        "supported by chemical principles and observed changes in pH and temperature."
    )
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7 = self.max_round
    all_thinking7 = [[] for _ in range(N_max_7)]
    all_answer7 = [[] for _ in range(N_max_7)]
    for r in range(N_max_7):
        for i, agent in enumerate(debate_agents_7):
            if r == 0:
                thinking7, answer7 = await agent(
                    [taskInfo, thinking6, answer6], debate_instruction_7, r, is_sub_task=True)
            else:
                input_infos_7 = [taskInfo, thinking6, answer6] + all_thinking7[r-1] + all_answer7[r-1]
                thinking7, answer7 = await agent(input_infos_7, debate_instruction_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, determining plausible reason, thinking: {thinking7.content}; answer: {answer7.content}")
            all_thinking7[r].append(thinking7)
            all_answer7[r].append(answer7)
    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7(
        [taskInfo] + all_thinking7[-1] + all_answer7[-1],
        "Sub-task 7: Make final decision on the most plausible reason for the reaction rate change.",
        is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing plausible reason, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
