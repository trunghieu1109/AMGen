async def forward_183(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Identify tert-butyl, ethoxy, and nitro substituents, assign their target positions (2-, 1-, 3- respectively), and characterize each substituent’s directing effect (electron-donating vs. withdrawing, strength, ortho/para vs. meta)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying substituents, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_instruction21 = "Sub-task 2.1: For each choice A–D, extract and enumerate the nine reaction steps in order, labeling them Step i through Step ix."
    cot_agent21 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc21 = {"subtask_id": "subtask_2.1", "instruction": cot_instruction21, "context": ["user query", "Sub-task 1 thinking", "Sub-task 1 answer"], "agent_collaboration": "CoT"}
    thinking21, answer21 = await cot_agent21([taskInfo, thinking1, answer1], cot_instruction21, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent21.id}, extracting steps, thinking: {thinking21.content}; answer: {answer21.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking21.content}; answer - {answer21.content}")
    subtask_desc21['response'] = {"thinking": thinking21, "answer": answer21}
    logs.append(subtask_desc21)
    cot_sc_instruction22 = "Sub-task 2.2: Classify each extracted step by reaction type (e.g., Friedel–Crafts alkylation, nitration, sulfonation, reduction, ether formation), producing a table of step→type for every choice."
    N = self.max_sc
    cot_agents22 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers22 = []
    thinking_map22 = {}
    answer_map22 = {}
    subtask_desc22 = {"subtask_id": "subtask_2.2", "instruction": cot_sc_instruction22, "context": ["user query","Sub-task 2.1 thinking","Sub-task 2.1 answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking22, answer22 = await cot_agents22[i]([taskInfo, thinking21, answer21], cot_sc_instruction22, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents22[i].id}, classifying steps, thinking: {thinking22.content}; answer: {answer22.content}")
        possible_answers22.append(answer22.content)
        thinking_map22[answer22.content] = thinking22
        answer_map22[answer22.content] = answer22
    answer22_content = Counter(possible_answers22).most_common(1)[0][0]
    thinking22 = thinking_map22[answer22_content]
    answer22 = answer_map22[answer22_content]
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking22.content}; answer - {answer22.content}")
    subtask_desc22['response'] = {"thinking": thinking22, "answer": answer22}
    logs.append(subtask_desc22)
    cot_sc_instruction31 = "Sub-task 3.1: Generate at least three plausible orders for installing tert-butyl, ethoxy, and nitro based on directing effects, anticipated yields, and mutual influence."
    N = self.max_sc
    cot_agents31 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers31 = []
    thinking_map31 = {}
    answer_map31 = {}
    subtask_desc31 = {"subtask_id": "subtask_3.1", "instruction": cot_sc_instruction31, "context": ["user query","Sub-task 1 answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking31, answer31 = await cot_agents31[i]([taskInfo, thinking1, answer1], cot_sc_instruction31, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents31[i].id}, generating orders, thinking: {thinking31.content}; answer: {answer31.content}")
        possible_answers31.append(answer31.content)
        thinking_map31[answer31.content] = thinking31
        answer_map31[answer31.content] = answer31
    answer31_content = Counter(possible_answers31).most_common(1)[0][0]
    thinking31 = thinking_map31[answer31_content]
    answer31 = answer_map31[answer31_content]
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking31.content}; answer - {answer31.content}")
    subtask_desc31['response'] = {"thinking": thinking31, "answer": answer31}
    logs.append(subtask_desc31)
    debate_instruction32 = "Sub-task 3.2: Debate the candidate sequences on regiochemical control, potential deactivation, and installation ease, citing directing strengths quantitatively."
    debate_agents32 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds32 = self.max_round
    all_thinking32 = [[] for _ in range(rounds32)]
    all_answer32 = [[] for _ in range(rounds32)]
    subtask_desc32 = {"subtask_id": "subtask_3.2", "instruction": debate_instruction32, "context": ["user query","Sub-task 3.1 answer"], "agent_collaboration": "Debate"}
    for r in range(rounds32):
        for i, agent in enumerate(debate_agents32):
            if r == 0:
                thinking32, answer32 = await agent([taskInfo, thinking31, answer31], debate_instruction32, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking31, answer31] + all_thinking32[r-1] + all_answer32[r-1]
                thinking32, answer32 = await agent(inputs, debate_instruction32, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating orders, thinking: {thinking32.content}; answer: {answer32.content}")
            all_thinking32[r].append(thinking32)
            all_answer32[r].append(answer32)
    final_consensus_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking32, answer32 = await final_consensus_agent([taskInfo] + all_thinking32[-1] + all_answer32[-1], "Sub-task 3.2: Summarize debate outputs.", is_sub_task=True)
    agents.append(f"Final Decision agent, summarizing debate, thinking: {thinking32.content}; answer: {answer32.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking32.content}; answer - {answer32.content}")
    subtask_desc32['response'] = {"thinking": thinking32, "answer": answer32}
    logs.append(subtask_desc32)
    cot_instruction33 = "Sub-task 3.3: Reach consensus on the optimal installation order (first, second, last) to maximize yield and regioselectivity for 2-(tert-butyl)-1-ethoxy-3-nitrobenzene."
    cot_agent33 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc33 = {"subtask_id": "subtask_3.3", "instruction": cot_instruction33, "context": ["user query","Sub-task 3.2 answer"], "agent_collaboration": "CoT"}
    thinking33, answer33 = await cot_agent33([taskInfo, thinking32, answer32], cot_instruction33, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent33.id}, reaching consensus, thinking: {thinking33.content}; answer: {answer33.content}")
    sub_tasks.append(f"Sub-task 3.3 output: thinking - {thinking33.content}; answer - {answer33.content}")
    subtask_desc33['response'] = {"thinking": thinking33, "answer": answer33}
    logs.append(subtask_desc33)
    cot_sc_instruction4 = "Sub-task 4: Map each choice’s classified nine-step sequence onto the consensus installation order from Sub-task 3.3, flagging any out-of-sequence or incompatible steps."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinking_map4 = {}
    answer_map4 = {}
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query","Sub-task 2.2 answer","Sub-task 3.3 answer"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking4, answer4 = await cot_agents4[i]([taskInfo, thinking22, answer22, thinking33, answer33], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, mapping sequences, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers4.append(answer4.content)
        thinking_map4[answer4.content] = thinking4
        answer_map4[answer4.content] = answer4
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_map4[answer4_content]
    answer4 = answer_map4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    cot_reflect_instruction5 = "Sub-task 5: Evaluate each mapped sequence for overall feasibility, considering cumulative yield, side reactions, isomer formation, and operational practicality."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5, "context": ["user query","Sub-task 4 answer"], "agent_collaboration": "Reflexion"}
    thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, assessing feasibility, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], "Please review the feasibility assessment and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refining assessment, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    debate_instruction6 = "Sub-task 6: Score each choice on ideal order match and feasibility criteria, then select the single best choice with justification."
    debate_agents6 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds6 = self.max_round
    all_thinking6 = [[] for _ in range(rounds6)]
    all_answer6 = [[] for _ in range(rounds6)]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": debate_instruction6, "context": ["user query","Sub-task 5 answer"], "agent_collaboration": "Debate"}
    for r in range(rounds6):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction6, r, is_sub_task=True)
            else:
                inputs6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(inputs6, debate_instruction6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, scoring choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_agent6 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_agent6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final selection of best choice with justification.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting best choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs