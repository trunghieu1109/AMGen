async def forward_7(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    stage1_context = """Definitions of genetic interaction types relevant for this analysis:
- Epistasis: When the phenotype of one gene masks or suppresses the effect of another gene.
- Redundancy: When two genes perform overlapping functions, and loss of both causes a stronger phenotype than either alone.
- Pleiotropy: When a single gene influences multiple phenotypic traits.
- Synthetic enhancement: When the combined mutation of two genes causes a more severe phenotype than expected from individual mutations.
"""

    cot_instruction_1 = (
        "Sub-task 1: Analyze the resistance levels of single mutants (g1, g2, g3) relative to the wild-type control to determine each gene's individual contribution to anthracnose resistance, "
        "identifying any gene whose knockout causes a strong loss of resistance. Use the experimental data provided in the query."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing single mutants resistance, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2a = (
        "Sub-task 2a: Determine epistatic relationships between gene pairs (G1-G2, G2-G3, G1-G3) by analyzing resistance levels of double mutants compared to single mutants and control, "
        "using multiple independent Chain-of-Thought reasoning runs to build consensus. Use the definitions of epistasis and experimental data."
    )
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "definitions of epistasis"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1, stage1_context], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining epistatic relationships, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a.content)
        thinkingmapping_2a[answer2a.content] = thinking2a
        answermapping_2a[answer2a.content] = answer2a
    answer2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking2a = thinkingmapping_2a[answer2a_content]
    answer2a = answermapping_2a[answer2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {
        "thinking": thinking2a,
        "answer": answer2a
    }
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_sc_instruction_2b = (
        "Sub-task 2b: Characterize the genetic interaction between G1 and G3 (e.g., redundancy, pleiotropy, synthetic enhancement) by performing a debate between agents using the resistance data and definitions provided, "
        "to reach a consensus on the most appropriate interaction type. Use outputs from Sub-task 2a as context."
    )
    debate_agents_2b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2b = self.max_round
    all_thinking2b = [[] for _ in range(N_max_2b)]
    all_answer2b = [[] for _ in range(N_max_2b)]
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "definitions of genetic interaction types"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2b):
        for i, agent in enumerate(debate_agents_2b):
            if r == 0:
                thinking2b, answer2b = await agent([taskInfo, thinking2a, answer2a, stage1_context], cot_sc_instruction_2b, r, is_sub_task=True)
            else:
                input_infos_2b = [taskInfo, thinking2a, answer2a, stage1_context] + all_thinking2b[r-1] + all_answer2b[r-1]
                thinking2b, answer2b = await agent(input_infos_2b, cot_sc_instruction_2b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, characterizing G1-G3 interaction, thinking: {thinking2b.content}; answer: {answer2b.content}")
            all_thinking2b[r].append(thinking2b)
            all_answer2b[r].append(answer2b)
    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2b, answer2b = await final_decision_agent_2b([taskInfo] + all_thinking2b[-1] + all_answer2b[-1], "Sub-task 2b: Make final decision on G1-G3 genetic interaction type.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding G1-G3 interaction, thinking: {thinking2b.content}; answer: {answer2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {
        "thinking": thinking2b,
        "answer": answer2b
    }
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_reflect_instruction_3 = (
        "Sub-task 3: Determine which gene is the likely transcription factor acting upstream by integrating single and double mutant resistance data, "
        "emphasizing genes whose knockout causes complete loss of resistance and whose double mutants do not further reduce resistance. Use outputs from Sub-tasks 1 and 2a."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2a, answer2a]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining transcription factor candidate, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "please review the transcription factor identification and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining transcription factor identification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Integrate validated conclusions from Sub-task 3 (transcription factor identification) and Sub-task 2b (G1-G3 interaction characterization) to select the correct multiple-choice answer, "
        "ensuring consistency with experimental data and prior analyses without re-deriving interaction logic. Use the provided multiple-choice options."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3, thinking2b, answer2b], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3, thinking2b, answer2b] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating final conclusions, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on gene interaction conclusion.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final conclusion, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
