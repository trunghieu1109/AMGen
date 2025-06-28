async def forward_139(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 1: Identify chemical nature and components, then identify liquid Y, gas W, and precipitate G

    # Sub-task 1: Identify chemical nature and composition of Substance X using Chain-of-Thought
    cot_instruction_1 = (
        "Sub-task 1: Identify the chemical nature and composition of Substance X based on clues: it incorporates a heavier isotope of one of its constituent elements, "
        "is used as a reagent in organic chemistry, and reacts violently with liquid Y producing gas W and precipitate G."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying Substance X, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Determine identity of liquid Y by analyzing reaction characteristics with Substance X using Chain-of-Thought
    cot_instruction_2 = (
        "Sub-task 2: Determine the identity of liquid Y by analyzing the reaction characteristics with Substance X, including violent reaction, gas W release, and precipitate G formation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, identifying liquid Y, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Identify gas W using Self-Consistency Chain-of-Thought (SC-CoT) due to complexity
    cot_sc_instruction_3 = (
        "Sub-task 3: Identify gas W by using the clue that its molecule contains the same number of neutrons and protons, "
        "and is released during the reaction of Substance X with liquid Y."
    )
    N_sc = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, identifying gas W, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    most_common_answer_3 = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[most_common_answer_3]
    answer3 = answermapping_3[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Identify precipitate G and analyze its thermal decomposition using Chain-of-Thought
    cot_instruction_4 = (
        "Sub-task 4: Identify precipitate G and analyze its thermal decomposition to release substance B, whose melting point is close to 277 K under normal conditions."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, identifying precipitate G and analyzing decomposition, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Analyze product from reaction with keto acid and determine isotopes involved

    # Sub-task 5: Analyze product formed from reaction of Substance X with keto acid using Chain-of-Thought
    cot_instruction_5 = (
        "Sub-task 5: Analyze the product formed from the reaction of Substance X with a certain keto acid, focusing on the product containing 2 oxygen atoms, "
        "to gain further insight into Substance X's composition and reactivity."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking1, answer1], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, analyzing keto acid reaction product, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Determine isotopes involved in Substance X using Reflexion (Chain-of-Thought + Critic)
    cot_reflect_instruction_6 = (
        "Sub-task 6: Determine the isotopes involved in Substance X, especially the heavier isotope mentioned, and identify all constituent elements including their isotopic variants, "
        "based on outputs from Sub-task 1 and Sub-task 5."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking1, answer1, thinking5, answer5]
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, determining isotopes, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking6, answer6],
                                                  "please review the isotope identification and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback on isotope identification, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback6])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, cot_reflect_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining isotope identification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Stage 3: Select lightest and heaviest elements and calculate cumulative atomic masses, then compare with choices

    # Sub-task 7: Select lightest and heaviest elements present in Substance X using Chain-of-Thought
    cot_instruction_7 = (
        "Sub-task 7: From the identified constituent elements and their isotopes in Substance X, select the lightest and heaviest elements present, "
        "considering all isotopic instances (both heavier and lighter isotopes) for each element."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting lightest and heaviest elements, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    # Sub-task 8: Calculate cumulative atomic masses of lightest and heaviest elements using Chain-of-Thought
    cot_instruction_8 = (
        "Sub-task 8: Calculate the cumulative atomic masses of the lightest and heaviest elements present in Substance X by summing the masses of all their isotopes (heavier and lighter) as identified."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, calculating cumulative atomic masses, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Subtask 8 answer: ", sub_tasks[-1])

    # Sub-task 9: Compare calculated cumulative atomic masses with provided choices using Debate
    debate_instruction_9 = (
        "Sub-task 9: Compare the calculated cumulative atomic masses with the provided answer choices (29, 25, 35, 31) to select the correct answer."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            input_infos_9 = [taskInfo, thinking8, answer8]
            if r > 0:
                input_infos_9 += all_thinking9[r-1] + all_answer9[r-1]
            thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing cumulative masses and selecting answer, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct cumulative atomic mass answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Subtask 9 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
